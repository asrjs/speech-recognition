"""Model-neutral input-source attribution for a failing stage boundary.

The technique this generalizes, proven on the Orukeet browser head gate:
when a downstream boundary fails, replay the component under test against the
official reference inputs, then replay it again with exactly one input
perturbed at a time. Comparing each single-input replay against the
reference-input replay shows which input actually carries the divergence,
instead of inferring it from the size of the observed error.

This module owns the control logic and the fail-closed semantics. It owns no
model knowledge: callers supply a callable that runs the component and
returns its outputs, plus the reference and candidate feed dicts. Graph
layout, decoding rules and tolerances stay with the family adapter.

Fail-closed rules, all verified by tests:

- Every input is perturbed. Missing a feed is an error, not a skipped case.
- A caller that changes an input silently is rejected by shape/dtype check.
- If no single input reproduces the observed failure, the control reports
  that attribution is UNRESOLVED. It never assigns blame to whichever input
  moved the metric most.
- Output comparison uses the full per-coordinate ``atol + rtol*|ref|`` gate.
  Dividing a single measured gain by ``atol`` alone understates the budget and
  is not permitted.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from tensor_parity import compare_tensors

# A replay returns the component's outputs as numpy arrays keyed by name.
RunStage = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def _as_float32(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != np.float32:
        raise ValueError(f"{label} must be float32, got {array.dtype}")
    if not array.all():
        finite = np.isfinite(array).all()
    else:  # pragma: no cover - fast path for dense tensors
        finite = np.isfinite(array).all()
    if not finite:
        raise ValueError(f"{label} contains non-finite values")
    return array


def _validate_feeds(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) ^ set(candidate))
        raise ValueError(f"feed sets differ: {missing}")
    if not reference:
        raise ValueError("attribution requires at least one input")
    for name in reference:
        left = np.asarray(reference[name])
        right = np.asarray(candidate[name])
        if left.shape != right.shape:
            raise ValueError(
                f"feed {name!r} shape differs: {left.shape} vs {right.shape}"
            )
        if left.dtype != right.dtype:
            raise ValueError(f"feed {name!r} dtype differs: {left.dtype} vs {right.dtype}")


def compare_stage_outputs(
    reference_outputs: Mapping[str, Any],
    candidate_outputs: Mapping[str, Any],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare every output the component produced, with no silent omission."""
    if set(reference_outputs) != set(candidate_outputs):
        missing = sorted(set(reference_outputs) ^ set(candidate_outputs))
        raise ValueError(f"output sets differ: {missing}")
    if not reference_outputs:
        raise ValueError("attribution requires at least one output")
    metrics: dict[str, Any] = {}
    passing = True
    for name in reference_outputs:
        result = compare_tensors(
            np.asarray(reference_outputs[name]),
            np.asarray(candidate_outputs[name]),
            atol=atol,
            rtol=rtol,
        )
        metrics[name] = result
        passing = passing and bool(result["allclose"])
    return {
        "allclose": passing,
        "atol": atol,
        "rtol": rtol,
        "outputs": metrics,
    }


def attribute_input_source(
    reference_feeds: Mapping[str, Any],
    candidate_feeds: Mapping[str, Any],
    run_stage: RunStage,
    *,
    atol: float,
    rtol: float,
    output_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Localize a downstream failure to the input that carries it.

    Runs the component three ways: on the reference inputs, on the candidate
    inputs, and once per input with that single input replaced by the
    candidate value. An input is implicated only when replaying it alone
    reproduces the observed failure.
    """
    if not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite non-negative number")
    if not isinstance(rtol, (int, float)) or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be a finite non-negative number")
    _validate_feeds(reference_feeds, candidate_feeds)

    deltas = {
        name: _as_float32(np.asarray(candidate_feeds[name]) - np.asarray(reference_feeds[name]), name)
        for name in reference_feeds
    }

    reference_run = dict(run_stage(dict(reference_feeds)))
    candidate_run = dict(run_stage(dict(candidate_feeds)))

    # A caller may attribute a subset of the component's outputs. Names must
    # exist on the runner's result, and the names actually selected are
    # recorded so a reader never assumes all outputs were compared.
    available = set(reference_run)
    selected = tuple(output_names) if output_names else tuple(available)
    if not selected or not set(selected) <= available:
        raise ValueError(
            f"output_names must be a subset of {sorted(available)}, got {sorted(selected)}"
        )
    outputs_are_subset = set(selected) != available

    observed = compare_stage_outputs(
        {n: reference_run[n] for n in selected},
        {n: candidate_run[n] for n in selected},
        atol=atol,
        rtol=rtol,
    )

    per_input: dict[str, Any] = {}
    implicated: list[str] = []
    for name in reference_feeds:
        # Substitute the candidate value directly. Rebuilding it as
        # reference + (candidate - reference) cancels catastrophically in
        # float32 (reference 1e10 with candidate 1 yields 0), which would
        # fabricate an UNRESOLVED result and hide a real attribution.
        perturbed = dict(reference_feeds)
        perturbed[name] = np.array(candidate_feeds[name], copy=True)
        replay = dict(run_stage(perturbed))
        single = compare_stage_outputs(
            {n: reference_run[n] for n in selected},
            {n: replay[n] for n in selected},
            atol=atol,
            rtol=rtol,
        )
        # Failing the gate alone is not enough. With interacting inputs a
        # single-input replay can fail while missing most of the observed
        # result, so the replay must also reproduce the candidate output.
        accounts = compare_stage_outputs(
            {n: replay[n] for n in selected},
            {n: candidate_run[n] for n in selected},
            atol=atol,
            rtol=rtol,
        )
        reproduces = not single["allclose"] and accounts["allclose"]
        if reproduces:
            implicated.append(name)
        per_input[name] = {
            "input_delta_max_abs_error": float(np.abs(deltas[name]).max())
            if deltas[name].size
            else 0.0,
            "fails_gate_against_reference": not single["allclose"],
            "reproduces_observed_output": bool(accounts["allclose"]),
            "reproduces_observed_failure": reproduces,
            "comparison": single,
            "agreement_with_candidate": accounts,
        }

    resolved = len(implicated) == 1 and not observed["allclose"]
    if observed["allclose"]:
        status = "BOUNDARY_PASSING"
        resolved = False
    elif resolved:
        status = "ATTRIBUTED_TO_SINGLE_INPUT"
    else:
        status = "ATTRIBUTION_UNRESOLVED"

    return {
        "status": status,
        "resolved": resolved,
        "implicated_inputs": implicated,
        "observed": observed,
        "per_input": per_input,
        "inputs_tested": sorted(reference_feeds),
        "outputs_tested": sorted(selected),
        "limitations": (
            "Scheduling and single-input replay attribution only. Not numerical "
            "feature parity, not a per-frame budget, and not evidence that any "
            "component is correct in general."
        ),
    }


def sweep_input_scale(
    reference_feeds: Mapping[str, Any],
    delta_feeds: Mapping[str, Any],
    run_stage: RunStage,
    *,
    scales: Sequence[float],
    atol: float,
    rtol: float,
    output_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Replay along a recorded input delta at several scales.

    Reports the largest tested scale that still passes. This is a directional
    bracket for one recorded direction only: it is not a boundary, and it is
    not a criterion for other frames or call directions.
    """
    _validate_feeds(reference_feeds, delta_feeds)
    if not scales:
        raise ValueError("scales must not be empty")

    reference_run = dict(run_stage(dict(reference_feeds)))
    available = set(reference_run)
    selected = tuple(output_names) if output_names else tuple(available)
    if not selected or not set(selected) <= available:
        raise ValueError(
            f"output_names must be a subset of {sorted(available)}, got {sorted(selected)}"
        )

    rows: list[dict[str, Any]] = []
    passing_scales: list[float] = []
    for scale in scales:
        # Convex form rather than reference + (delta - reference)*scale. The
        # latter cancels catastrophically in float32: with reference 1e10 and
        # delta 1, scale 1.0 yields 0 instead of 1. This form is exact at both
        # endpoints.
        feeds = {
            name: np.asarray(reference_feeds[name], dtype=np.float32) * (1.0 - float(scale))
            + np.asarray(delta_feeds[name], dtype=np.float32) * float(scale)
            for name in reference_feeds
        }
        replay = dict(run_stage(feeds))
        result = compare_stage_outputs(
            {n: reference_run[n] for n in selected},
            {n: replay[n] for n in selected},
            atol=atol,
            rtol=rtol,
        )
        if result["allclose"]:
            passing_scales.append(float(scale))
        rows.append(
            {
                "scale": float(scale),
                "allclose": bool(result["allclose"]),
                "mismatching_outputs": sorted(
                    n for n in selected if not result["outputs"][n]["allclose"]
                ),
                "first_divergent_index": {
                    n: result["outputs"][n]["first_divergent_index"] for n in selected
                },
                "max_abs_error": {
                    n: result["outputs"][n]["max_abs_error"] for n in selected
                },
                "mismatch_count": {
                    n: result["outputs"][n]["mismatch_count"] for n in selected
                },
            }
        )

    largest_passing = max(passing_scales) if passing_scales else None
    return {
        "rows": rows,
        "largest_passing_scale": largest_passing,
        "atol": atol,
        "rtol": rtol,
        "limitations": (
            "Directional bracket along one recorded input delta at the replayed "
            "boundary. Not a per-frame budget and not a necessary accuracy bound."
        ),
    }