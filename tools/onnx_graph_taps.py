"""Expose named intermediate tensors in a fresh self-contained diagnostic graph.

Additional outputs can change optimizer/provider partitioning. This is not a
production graph transformation or an acceptance substitute.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import onnx

from onnx_artifacts import iter_tensors


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_graph_taps(model, names):
    if not names or len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError('Nonempty unique tap names required')
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in iter_tensors(model)):
        raise ValueError('Only self-contained graphs are supported; external weights must not be copied implicitly')
    existing = {v.name for v in model.graph.output}
    produced = {name for node in model.graph.node for name in node.output}
    if any(name in existing or name not in produced for name in names):
        raise ValueError('Tap must be an intermediate node output, not an existing output')
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    metadata = {v.name: v for v in (*inferred.graph.value_info, *inferred.graph.output)}
    if any(name not in metadata or not metadata[name].type.HasField('tensor_type')
           or not metadata[name].type.tensor_type.elem_type
           or not metadata[name].type.tensor_type.HasField('shape') for name in names):
        raise ValueError('Inferred tensor dtype/rank required for every tap')
    result = copy.deepcopy(model)
    for name in names:
        result.graph.output.append(metadata[name])
    onnx.checker.check_model(result)
    # Shape inference is used only for metadata: preserve every original field.
    control = copy.deepcopy(result)
    del control.graph.output[len(model.graph.output):]
    if control.SerializeToString() != model.SerializeToString():
        raise ValueError('Diagnostic transformation changed more than appended outputs')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--tap', action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source_hash = sha256(args.source)
    if source_hash != args.expected_sha256:
        raise ValueError('Source graph hash differs')
    model = onnx.load(str(args.source), load_external_data=False)
    diagnostic = add_graph_taps(model, args.tap)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / 'diagnostic.onnx'
    onnx.save(diagnostic, str(output))
    onnx.checker.check_model(str(output))
    receipt = {'schema_version': 1, 'status': 'DIAGNOSTIC_GRAPH_CREATED_UNVALIDATED',
               'source': str(args.source.resolve()), 'source_sha256': source_hash,
               'graph': str(output.resolve()), 'graph_sha256': sha256(output),
               'tool_sha256': sha256(__file__), 'onnx_version': onnx.__version__,
               'taps': args.tap, 'only_appended_outputs': True,
               'warning': 'Additional outputs can change optimizer/provider partitioning'}
    with (args.output_dir / 'tap-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
