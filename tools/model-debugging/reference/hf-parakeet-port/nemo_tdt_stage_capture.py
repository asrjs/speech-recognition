"""Capture CPU NeMo greedy-batch TDT boundaries without replacing its decoder.

Intentionally specific to NeMo's projected-joint label-looping path. Numerical
captures are reference evidence, not a claim that an export/browser port passes.
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch


class NemoTdtStageCapture:
    def __init__(self, model: Any, max_calls: int = 4096):
        if max_calls < 1:
            raise ValueError("max_calls must be positive")
        self.model = model
        self.max_calls = max_calls
        self.stages: dict[str, dict] = {}
        self.calls: list[dict] = []
        self.counts = {"predictor": 0, "joint": 0, "encoder_projection": 0, "predictor_projection": 0}
        self._cleanup = ExitStack()

    def record(self, name: str, value: torch.Tensor) -> None:
        if name in self.stages:
            raise RuntimeError(f"Duplicate capture stage: {name}; capture one audio item per run")
        if not isinstance(value, torch.Tensor) or not value.numel():
            raise RuntimeError(f"Empty/non-tensor capture stage: {name}")
        if value.device.type != "cpu":
            raise RuntimeError("Stage capture requires CPU reference execution (no CUDA graph hooks)")
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite capture stage: {name}")
        snapshot = value.detach().contiguous().clone()
        self.stages[name] = {
            "shape": list(snapshot.shape),
            "dtype": str(snapshot.dtype).removeprefix("torch."),
            "data": snapshot.reshape(-1).tolist(),
        }

    def record_state(self, prefix: str, state: Any) -> None:
        if state is None:
            return
        if not isinstance(state, (list, tuple)) or len(state) != 2:
            raise RuntimeError("Expected the NeMo LSTM hidden/cell pair")
        for suffix, tensor in zip(("h", "c"), state):
            self.record(f"{prefix}_{suffix}", tensor)

    def model_input_contract(self, sample_rate: int) -> dict:
        waveform = self.stages.get("waveform", {})
        length = self.stages.get("waveform_length", {})
        shape = waveform.get("shape", [])
        lengths = length.get("data", [])
        if (type(sample_rate) is not int or sample_rate < 1
                or len(shape) != 2 or shape[0] != 1 or shape[1] < 1
                or waveform.get("dtype") != "float32"
                or len(waveform.get("data", [])) != shape[1]
                or length.get("shape") != [1]
                or length.get("dtype") not in ("int32", "int64")
                or len(lengths) != 1 or type(lengths[0]) is not int
                or not 0 < lengths[0] <= shape[1]):
            raise RuntimeError("Invalid model-input waveform/rate/length contract")
        return {
            "boundary": "preprocessor_input_signal",
            "sample_rate": sample_rate, "channels": 1, "dtype": "float32",
            "encoding": "f32le", "shape": shape.copy(),
            "samples": shape[1], "valid_samples": lengths[0],
            "right_padding_samples": shape[1] - lengths[0],
            "sha256": hashlib.sha256(
                np.asarray(waveform["data"], dtype="<f4").tobytes(order="C"),
            ).hexdigest(),
        }

    def _hook_waveform(self, module: Any) -> None:
        signature = inspect.signature(module.forward)

        def capture_input(_module, args, kwargs):
            bound = signature.bind(*args, **kwargs)
            self.record("waveform", bound.arguments["input_signal"])
            self.record("waveform_length", bound.arguments["length"])

        handle = module.register_forward_pre_hook(capture_input, with_kwargs=True)
        self._cleanup.callback(handle.remove)

    def _hook(self, module: Any, prefix: str) -> None:
        def capture_output(_module, _inputs, output):
            self.record(prefix, output[0])
            self.record(f"{prefix}_length", output[1])

        handle = module.register_forward_hook(capture_output)
        self._cleanup.callback(handle.remove)

    def _wrap(self, owner: Any, name: str, kind: str) -> None:
        original = getattr(owner, name)
        signature = inspect.signature(original)
        had_override = name in owner.__dict__
        previous_override = owner.__dict__.get(name)

        def restore():
            if had_override:
                setattr(owner, name, previous_override)
            else:
                delattr(owner, name)

        def capture_call(*args, **kwargs):
            index = self.counts[kind]
            if index >= self.max_calls:
                raise RuntimeError(f"Capture call limit exceeded: {kind}; no truncated PASS")
            self.counts[kind] += 1
            prefix = f"{kind}/{index:04d}"
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            call = {"kind": kind, "index": index, "stage_prefix": prefix}
            self.calls.append(call)
            if kind == "predictor":
                labels, state = bound.arguments["y"], bound.arguments["state"]
                call.update({"labels_missing": labels is None, "state_missing": state is None,
                             "add_sos": bound.arguments["add_sos"]})
                if labels is not None:
                    self.record(f"{prefix}/labels", labels)
                self.record_state(f"{prefix}/input_state", state)
            else:
                for arg_name, value in bound.arguments.items():
                    if isinstance(value, torch.Tensor):
                        self.record(f"{prefix}/input_{arg_name}", value)
            output = original(*args, **kwargs)
            if kind == "predictor":
                self.record(f"{prefix}/output", output[0])
                self.record_state(f"{prefix}/output_state", output[1])
            else:
                self.record(f"{prefix}/output", output)
            return output

        setattr(owner, name, capture_call)
        self._cleanup.callback(restore)

    def __enter__(self):
        cfg = self.model.cfg.decoding
        if cfg.strategy != "greedy_batch" or cfg.model_type != "tdt":
            raise RuntimeError("Capture supports official greedy_batch TDT only")
        try:
            self._hook_waveform(self.model.preprocessor)
            self._hook(self.model.preprocessor, "features")
            self._hook(self.model.encoder, "encoder")
            self._wrap(self.model.decoder, "predict", "predictor")
            self._wrap(self.model.joint, "project_encoder", "encoder_projection")
            self._wrap(self.model.joint, "project_prednet", "predictor_projection")
            self._wrap(self.model.joint, "joint_after_projection", "joint")
        except Exception:
            self._cleanup.close()
            raise
        return self

    def __exit__(self, *exception):
        self._cleanup.close()
        return False

    def validate_complete(self) -> None:
        required = ("waveform", "waveform_length", "features", "features_length", "encoder", "encoder_length")
        if any(name not in self.stages for name in required) or any(count == 0 for count in self.counts.values()):
            raise RuntimeError("Incomplete official TDT capture; no numerical reference produced")
