"""Give every LayerNormalization an explicit 1-D zero bias.

The exported Moonshine graphs emit LayerNormalization with two inputs, so ONNX
Runtime's WebGPU fusion to SkipLayerNormalization has no bias to read and
synthesises one with the wrong rank, which the WebGPU kernel rejects with
"Beta must be 1D".

Adding an explicit zero bias is numerically neutral, because an absent bias and a
zero bias mean the same thing; it only gives the fusion a 1-D tensor to bind to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def patch(path: Path) -> dict:
    model = onnx.load(str(path), load_external_data=False)
    initializers = {entry.name: entry for entry in model.graph.initializer}
    added, already = 0, 0
    for node in model.graph.node:
        if node.op_type != "LayerNormalization" or len(node.input) >= 3:
            continue
        scale = initializers.get(node.input[1])
        if scale is None or len(scale.dims) != 1:
            continue
        name = f"{node.name or node.input[1]}_zero_bias"
        if name in initializers:
            already += 1
        else:
            tensor = numpy_helper.from_array(np.zeros(scale.dims, dtype=np.float32), name)
            model.graph.initializer.append(tensor)
            added += 1
        node.input.append(name)
    onnx.save(model, str(path))
    return {"path": str(path), "biasesAdded": added, "alreadyPresent": already}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graphs", nargs="+", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps([patch(path) for path in args.graphs], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
