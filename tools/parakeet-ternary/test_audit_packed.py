"""Synthetic, offline preflight controls; no cached model is required."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from audit_packed import DIGIT_ORDER, WEIGHT_RULE, audit, check_identity, main, read_verified


class PackedAuditTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "format": "thrush-ternary-v2", "names": "hf",
            "packing": {"base": 3, "code_offset": 1, "elements_per_byte": 5,
                        "row_bytes": "ceil(in/5)", "row_major": True,
                        "digit_order": DIGIT_ORDER},
            "weight_rule": WEIGHT_RULE,
            "quant": {"mode": "ternary", "group_size": 4},
            "quantized_modules": [{"name": "layer", "out_features": 2,
                                   "in_features": 6, "group_size": 4}],
            "n_quantized_params": 12,
        }
        self.config = {"ternary_group_size": 4, "ternary_modules": ["layer"]}
        self.tensors = {"layer.qweight": np.zeros((2, 2), dtype=np.uint8),
                        "layer.scales": np.ones((2, 2), dtype=np.float16)}

    def run_audit(self):
        return audit(self.manifest, self.config, self.tensors)

    def test_valid_small_manifest(self):
        result = self.run_audit()
        self.assertEqual(result["status"], "PASS_PACKED_FORMAT_ONLY")
        self.assertEqual(result["quantizedParams"], 12)
        self.assertEqual(result["estimatedTwoBitCodeBytes"], 4)

    def test_invalid_byte(self):
        self.tensors["layer.qweight"][0, 0] = 243
        with self.assertRaisesRegex(ValueError, "invalid packed ternary byte"):
            self.run_audit()

    def test_nonzero_padding(self):
        self.tensors["layer.qweight"][0, -1] = 3
        with self.assertRaisesRegex(ValueError, "nonzero padding"):
            self.run_audit()

    def test_nonfinite_scale(self):
        self.tensors["layer.scales"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_audit()

    def test_duplicate_modules(self):
        self.manifest["quantized_modules"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.run_audit()

    def test_missing_and_unlisted_tensors(self):
        for mode in ("missing", "extra"):
            with self.subTest(mode=mode):
                tensors = dict(self.tensors)
                if mode == "missing":
                    del tensors["layer.scales"]
                else:
                    tensors["extra.qweight"] = np.zeros((1, 1), dtype=np.uint8)
                with self.assertRaisesRegex(ValueError, "missing or unlisted"):
                    audit(self.manifest, self.config, tensors)

    def test_dtype_and_shape(self):
        for key, value in (("layer.qweight", np.zeros((2, 2), dtype=np.int8)),
                           ("layer.scales", np.ones((2, 2), dtype=np.float32)),
                           ("layer.scales", np.ones((2, 1), dtype=np.float16))):
            with self.subTest(key=key, shape=value.shape, dtype=value.dtype):
                with self.assertRaisesRegex(ValueError, "dtype/shape"):
                    audit(self.manifest, self.config, {**self.tensors, key: value})

    def test_config_mismatch(self):
        self.config["ternary_modules"] = ["other"]
        with self.assertRaisesRegex(ValueError, "module mismatch"):
            self.run_audit()

    def test_wrong_parameter_count(self):
        self.manifest["n_quantized_params"] = 13
        with self.assertRaisesRegex(ValueError, "parameter count"):
            self.run_audit()

    def test_invalid_group(self):
        self.manifest["quant"]["group_size"] = 0
        with self.assertRaisesRegex(ValueError, "quantization group"):
            self.run_audit()

    def test_changed_semantic_contract(self):
        for field in ("digit_order", "weight_rule"):
            with self.subTest(field=field):
                manifest = {**self.manifest, "packing": dict(self.manifest["packing"])}
                if field == "digit_order":
                    manifest["packing"][field] = "most significant first"
                else:
                    manifest[field] = "code + 1"
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    audit(manifest, self.config, self.tensors)

    def test_identity_required_and_matched(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"fixture")
            digest = hashlib.sha256(b"fixture").hexdigest()
            self.assertEqual(check_identity(path, digest.upper()), digest)
            for wrong in ("", "not-a-hash", "0" * 64):
                with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                    check_identity(path, wrong)

    def test_cli_consumes_verified_snapshots_without_reopening_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b""
            header = {}
            for name, tensor in self.tensors.items():
                raw = tensor.tobytes()
                header[name] = {"dtype": "U8" if tensor.dtype == np.uint8 else "F16",
                                "shape": list(tensor.shape),
                                "data_offsets": [len(payload), len(payload) + len(raw)]}
                payload += raw
            encoded = json.dumps(header).encode()
            inputs = {"archive": struct.pack("<Q", len(encoded)) + encoded + payload,
                      "manifest": json.dumps(self.manifest).encode(),
                      "config": json.dumps(self.config).encode()}
            args = ["audit_packed.py"]
            for role, data in inputs.items():
                path = root / role
                path.write_bytes(data)
                args.extend(["--" + role, str(path), "--" + role + "-sha256",
                             hashlib.sha256(data).hexdigest()])
            output = io.StringIO()
            with patch("sys.argv", args), patch("sys.stdout", output), patch(
                "pathlib.Path.read_text", side_effect=AssertionError("unverified path reopen")
            ):
                self.assertEqual(main(), 0)
            self.assertEqual(json.loads(output.getvalue())["status"], "PASS_PACKED_FORMAT_ONLY")

    def test_verified_snapshot_survives_file_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input"
            path.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            snapshot, verified_digest = read_verified(path, digest)
            path.write_bytes(b"replacement")
            self.assertEqual(snapshot, b"original")
            self.assertEqual(hashlib.sha256(snapshot).hexdigest(), verified_digest)


if __name__ == "__main__":
    unittest.main()
