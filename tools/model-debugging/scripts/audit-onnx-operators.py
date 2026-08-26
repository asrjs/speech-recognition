#!/usr/bin/env python3
"""Emit a lightweight ONNX operator inventory without loading external data."""

from __future__ import annotations

import collections
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"status": "unavailable", "error": "expected one ONNX path"}))
        return 2
    try:
        import onnx

        model = onnx.load(sys.argv[1], load_external_data=False)
        operators = collections.Counter(node.op_type for node in model.graph.node)
        domains = collections.Counter(node.domain or "ai.onnx" for node in model.graph.node)
        hints = []
        if any(name in operators for name in ("ConvInteger", "MatMulInteger", "QLinearConv")):
            hints.append(
                "integer arithmetic operators require explicit WASM/WebGPU provider verification"
            )
        if any(name in operators for name in ("Loop", "If", "Scan", "GridSample", "Unique")):
            hints.append("advanced/control-flow operators require explicit ORT Web verification")
        if any(domain != "ai.onnx" for domain in domains):
            hints.append("non-default operator domains require provider and package verification")
        print(
            json.dumps(
                {
                    "status": "ok",
                    "ir_version": model.ir_version,
                    "opset_import": [
                        {"domain": item.domain or "ai.onnx", "version": item.version}
                        for item in model.opset_import
                    ],
                    "operator_count": len(model.graph.node),
                    "operators": dict(sorted(operators.items())),
                    "domains": dict(sorted(domains.items())),
                    "provider_risk_hints": hints,
                }
            )
        )
        return 0
    except Exception as error:  # noqa: BLE001 - report audit failure as JSON
        print(json.dumps({"status": "unavailable", "error": str(error)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
