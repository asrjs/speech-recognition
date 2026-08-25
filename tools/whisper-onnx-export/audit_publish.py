#!/usr/bin/env python3
"""Publish-readiness audit for Whisper 4-graph ONNX export directories.

Usage:
  python audit_publish.py /path/to/export-dir [--variant fp32|fp16|int8-dynamic]

Checks:
  1. Directory layout — no tensor-named files in root or variant dirs.
  2. External data correctness — locations match manifest, offsets valid.
  3. ONNX validation — path-based checker for every graph.
  4. ORT load — every graph loads via InferenceSession.
  5. Manifest consistency — paths match ONNX internal locations.
  6. Variant self-containment — no cross-variant references.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import onnx
import onnxruntime as ort

# ---- Forbidden filename patterns (tensor-named external data files) ----
_BAD_PATTERNS = [
    "encoder.layers.",
    "decoder.layers.",
    "model.encoder.",
    "model.decoder.",
    "encoder.conv",
    "encoder.layer_norm",
    "decoder.embed",
    "onnx__",
]

_EXPECTED_GRAPH_FILES = [
    "encoder_model.onnx",
    "decoder_init.onnx",
    "decoder_step.onnx",
    "decoder_align.onnx",
]

_EXPECTED_DATA_FILES = [
    "encoder_model.onnx.data",
    "decoder_init.onnx.data",
    "decoder_step.onnx.data",
    "decoder_align.onnx.data",
]


def _is_bad_filename(name: str) -> bool:
    """Return True if this looks like a per-weight external data file."""
    for pat in _BAD_PATTERNS:
        if name.startswith(pat):
            return True
    return False


def audit_directory(
    root: Path,
    *,
    variant: Optional[str] = None,
    smoke_decode: bool = False,
    reference_json: Optional[Path] = None,
) -> Tuple[int, int]:
    """Run full publish audit. Returns (passes, failures)."""
    passes = 0
    failures = 0
    issues: List[str] = []

    def check(condition: bool, msg: str) -> None:
        nonlocal passes, failures
        if condition:
            passes += 1
            print(f"  ✓ {msg}")
        else:
            failures += 1
            print(f"  ✗ {msg}")
            issues.append(msg)

    print(f"=== Audit: {root} ===\n")

    # ---- 1. Find variant directories ----
    variant_dirs: List[Path] = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "manifest.json").exists():
            variant_dirs.append(d)

    if variant:
        variant_dirs = [d for d in variant_dirs if d.name == variant]
        if not variant_dirs:
            print(f"  ✗ No variant directory '{variant}' found")
            return passes, 1

    if not variant_dirs:
        # Flat layout: root is the variant dir
        if (root / "manifest.json").exists():
            variant_dirs = [root]
        else:
            print("  ✗ No manifest.json found; not a valid export directory")
            return passes, 1

    for vdir in variant_dirs:
        print(f"\n--- Variant: {vdir.name} ---\n")
        _audit_variant_dir(vdir, check, smoke_decode, reference_json)

    # ---- Root-level check ----
    print(f"\n--- Root-level check ---\n")
    root_bad = []
    for f in root.iterdir():
        if f.is_file() and _is_bad_filename(f.name):
            root_bad.append(f.name)
    check(len(root_bad) == 0, f"No tensor-named files in root ({len(root_bad)} found)")
    for b in root_bad:
        print(f"    BAD: {b}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Total: {passes} passes, {failures} failures")
    if failures:
        print(f"  Issues:")
        for i in issues:
            print(f"    - {i}")
    print(f"{'='*60}")

    return passes, failures


def _audit_variant_dir(
    vdir: Path,
    check,
    smoke_decode: bool = False,
    reference_json: Optional[Path] = None,
):
    """Audit a single variant directory."""

    # ---- 1a. Manifest parsing ----
    manifest_path = vdir / "manifest.json"
    check(manifest_path.exists(), "manifest.json exists")

    manifest: Dict[str, Any] = {}
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
        check(manifest.get("format") == "whisper-browser-self-export-v1", "manifest format correct")
    except Exception as e:
        check(False, f"Failed to parse manifest: {e}")
        return

    artifacts = manifest.get("artifacts", {})

    # A decoder_align graph without the causal teacher-forced export marker is
    # loadable but only supports the runtime's generated timestamp fallback.
    # Keep that legacy state visible in development, but reject it as a
    # publish-ready timestamp artifact.
    has_decoder_align = "decoder_align" in artifacts or (vdir / "decoder_align.onnx").exists()
    if has_decoder_align:
        alignment_export = manifest.get("alignment_export")
        check(
            isinstance(alignment_export, dict)
            and alignment_export.get("causal_self_attention") is True,
            "alignment_export.causal_self_attention is true",
        )
        check(
            isinstance(alignment_export, dict)
            and alignment_export.get("attention_values") == "logits",
            "alignment_export.attention_values is logits",
        )
        check(
            isinstance(alignment_export, dict)
            and alignment_export.get("attention_layout") == "selected_heads",
            "alignment_export.attention_layout is selected_heads",
        )

    # ---- 1b. No tensor-named files ----
    bad_files = [f.name for f in vdir.iterdir() if f.is_file() and _is_bad_filename(f.name)]
    check(len(bad_files) == 0, f"No tensor-named files in {vdir.name}/ ({len(bad_files)} found)")
    for b in bad_files:
        print(f"    BAD: {b}")

    # ---- 1c. All expected graph files present ----
    for gname in _EXPECTED_GRAPH_FILES:
        gpath = vdir / gname
        check(gpath.exists(), f"{vdir.name}/{gname} exists")
        if not gpath.exists():
            continue

        # ---- 1d. Check external data ----
        ext_files = _get_external_data_files(gpath)
        if ext_files:
            for ext_loc, ext_path in ext_files.items():
                check(ext_path.exists(), f"  {gname}: external data {ext_loc} exists")
                if ext_path.exists():
                    actual_sz = ext_path.stat().st_size
                    check(actual_sz > 0, f"  {gname}: {ext_loc} non-empty ({actual_sz / 1024 / 1024:.1f} MB)")

    # ---- 2. Manifest ↔ ONNX location agreement ----
    for art_key, art_val in artifacts.items():
        if not isinstance(art_val, dict):
            continue
        onnx_file = art_val.get("file", "")
        gpath = vdir / onnx_file
        if not gpath.exists():
            continue

        manifest_ext = art_val.get("externalData", [])
        onnx_ext_locs = _get_external_data_locations(gpath)

        # Manifest entries must match ONNX internal locations
        for mentry in manifest_ext:
            mpath = mentry.get("path", "")
            # Strip ./ prefix if present
            if mpath.startswith("./"):
                mpath = mpath[2:]
            found = False
            for oloc in onnx_ext_locs:
                if oloc.endswith(mpath) or mpath.endswith(oloc) or oloc == mpath or f"./{oloc}" == mentry.get("path"):
                    found = True
                    # Check size info
                    if mentry.get("sizeBytes"):
                        data_file = vdir / oloc
                        if data_file.exists():
                            check(
                                data_file.stat().st_size == mentry["sizeBytes"],
                                f"  {art_key}: manifest sizeBytes {mentry['sizeBytes']} == actual {data_file.stat().st_size}",
                            )
                    break
            if onnx_ext_locs:
                check(found, f"  {art_key}: manifest path '{mpath}' matches ONNX location")

        # ONNX locations must have manifest entries
        seen_onnx_locs: set[str] = set()
        for oloc in onnx_ext_locs:
            if oloc in seen_onnx_locs:
                continue
            seen_onnx_locs.add(oloc)
            found_manifest = any(
                mentry.get("path", "").replace("./", "") in oloc or oloc in mentry.get("path", "").replace("./", "")
                for mentry in manifest_ext
            )
            check(found_manifest, f"  {art_key}: ONNX location '{oloc}' has manifest entry")

    # ---- 3. External data offset/length audit ----
    for gname in _EXPECTED_GRAPH_FILES:
        gpath = vdir / gname
        if not gpath.exists():
            continue
        _audit_external_data_offsets(gpath, vdir, check)

    # ---- 4. ONNX path-based checker ----
    for gname in _EXPECTED_GRAPH_FILES:
        gpath = vdir / gname
        if not gpath.exists():
            continue
        try:
            onnx.checker.check_model(str(gpath))
            check(True, f"  {gname}: onnx.checker.check_model(path) OK")
        except Exception as e:
            check(False, f"  {gname}: onnx.checker.check_model failed: {e}")

    # ---- 5. ORT load ----
    for gname in _EXPECTED_GRAPH_FILES:
        gpath = vdir / gname
        if not gpath.exists():
            continue
        try:
            sess = ort.InferenceSession(str(gpath), providers=['CPUExecutionProvider'])
            check(True, f"  {gname}: ORT load OK ({len(sess.get_inputs())} in, {len(sess.get_outputs())} out)")
        except Exception as e:
            check(False, f"  {gname}: ORT load failed: {e}")

    # ---- 6. Self-containment ----
    check((vdir / "config.json").exists(), f"{vdir.name}/config.json exists")
    check((vdir / "tokenizer.json").exists(), f"{vdir.name}/tokenizer.json exists")

    # ---- 7. External data files don't reference paths outside variant dir ----
    for gname in _EXPECTED_GRAPH_FILES:
        gpath = vdir / gname
        if not gpath.exists():
            continue
        for loc in _get_external_data_locations(gpath):
            check(
                "../" not in loc,
                f"  {gname}: location '{loc}' is relative (no ../ escapes)",
            )

    # ---- 8. SHA256 checksums ----
    print(f"\n  SHA256 checksums:")
    for gname in _EXPECTED_GRAPH_FILES + _EXPECTED_DATA_FILES:
        fpath = vdir / gname
        if not fpath.exists():
            continue
        sha = hashlib.sha256(fpath.read_bytes()).hexdigest()
        sz = fpath.stat().st_size
        print(f"    {sz / 1024 / 1024:8.1f} MB  {sha[:16]}...  {vdir.name}/{gname}")

    # ---- 9. Smoke decode (optional) ----
    if smoke_decode and reference_json:
        _run_smoke_decode(vdir, reference_json, check)


def _get_external_data_files(graph_path: Path) -> Dict[str, Path]:
    """Map ONNX internal location → actual Path for external data."""
    result: Dict[str, Path] = {}
    try:
        model = onnx.load(str(graph_path), load_external_data=False)
    except Exception:
        return result
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                loc = entry.value
                # Resolve relative to graph dir
                resolved = (graph_path.parent / loc).resolve()
                result[loc] = resolved
                break
    return result


def _get_external_data_locations(graph_path: Path) -> List[str]:
    """Return all external_data 'location' values from an ONNX graph."""
    locs: List[str] = []
    try:
        model = onnx.load(str(graph_path), load_external_data=False)
    except Exception:
        return locs
    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        for entry in init.external_data:
            if entry.key == "location":
                locs.append(entry.value)
                break
    return locs


def _audit_external_data_offsets(graph_path: Path, vdir: Path, check):
    """Verify offset+length <= file_size and no overlaps."""
    try:
        model = onnx.load(str(graph_path), load_external_data=False)
    except Exception:
        return

    data_files: Dict[str, int] = {}  # location → file_size
    spans: List[Tuple[str, int, int]] = []  # (location, offset, offset+length)

    for init in model.graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        loc = offset = length = None
        for entry in init.external_data:
            if entry.key == "location":
                loc = entry.value
            elif entry.key == "offset":
                offset = int(entry.value)
            elif entry.key == "length":
                length = int(entry.value)
        if loc is None:
            continue
        data_path = vdir / loc
        if data_path.exists() and loc not in data_files:
            data_files[loc] = data_path.stat().st_size
        if offset is not None and length is not None:
            spans.append((loc, offset, offset + length))

    # Check offset+length <= file_size
    for loc, start, end in spans:
        file_sz = data_files.get(loc)
        if file_sz:
            check(
                end <= file_sz,
                f"  {graph_path.name}: {loc} offset {start}+{end-start} <= {file_sz}",
            )

    # Check single-file case: all spans in contiguous order with no gaps
    if len(data_files) == 1:
        loc = list(data_files.keys())[0]
        loc_spans = sorted([(s, e) for l, s, e in spans if l == loc])
        if loc_spans:
            # Check contiguous (no gaps between spans)
            gaps = 0
            for i in range(1, len(loc_spans)):
                prev_end = loc_spans[i - 1][1]
                curr_start = loc_spans[i][0]
                if curr_start != prev_end:
                    gaps += 1
            check(
                gaps == 0,
                f"  {graph_path.name}: {loc} spans contiguous ({len(loc_spans)} spans, {gaps} gaps)",
            )
            # Check first starts at 0
            check(
                loc_spans[0][0] == 0,
                f"  {graph_path.name}: {loc} first span starts at 0",
            )


def _run_smoke_decode(vdir: Path, reference_json: Path, check):
    """Minimal smoke decode using ORT."""
    import numpy as np
    from pathlib import Path as P

    ref_path = P(reference_json)
    if not ref_path.exists():
        print(f"  SKIP smoke: reference JSON not found at {ref_path}")
        return

    try:
        with open(ref_path) as f:
            ref = json.load(f)
    except Exception as e:
        print(f"  SKIP smoke: failed to load reference: {e}")
        return

    try:
        enc_sess = ort.InferenceSession(str(vdir / "encoder_model.onnx"), providers=['CPUExecutionProvider'])
        init_sess = ort.InferenceSession(str(vdir / "decoder_init.onnx"), providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"  SKIP smoke: session creation failed: {e}")
        return

    # Run encoder with a dummy mel
    mel = np.random.randn(1, 80, 3000).astype(np.float32)
    enc_out = enc_sess.run(None, {"input_features": mel})
    print(f"  Smoke encoder: output shape {enc_out[0].shape}")
    check(enc_out[0].shape[0] == 1, "Smoke encoder batch dim OK")

    # Run decoder_init with dummy input_ids + encoder output
    input_ids = np.array([[50258, 50259, 50359, 50363]], dtype=np.int64)
    init_out = init_sess.run(None, {"input_ids": input_ids, "encoder_hidden_states": enc_out[0]})
    print(f"  Smoke decoder_init: {len(init_out)} outputs")
    check(len(init_out) > 0, "Smoke decoder_init produces outputs")

    print(f"  Smoke decode: OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Publish-readiness audit for Whisper ONNX exports")
    parser.add_argument("directory", type=str, help="Path to export directory")
    parser.add_argument("--variant", type=str, default=None, help="Specific variant to audit (fp32, fp16, int8-dynamic)")
    parser.add_argument("--smoke", action="store_true", help="Run minimal smoke decode")
    parser.add_argument("--reference", type=str, default=None, help="Path to reference JSON for smoke decode")
    args = parser.parse_args()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"  ✗ Not a directory: {root}")
        sys.exit(1)

    passes, failures = audit_directory(
        root,
        variant=args.variant,
        smoke_decode=args.smoke,
        reference_json=Path(args.reference) if args.reference else None,
    )

    sys.exit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
