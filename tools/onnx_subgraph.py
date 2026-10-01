"""Extract a hash-pinned self-contained diagnostic boundary, never a production port.

Extraction can change optimizer/provider partitioning. Validate matched-input
and full-graph output controls separately before trusting diagnostic evidence.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import onnx

from onnx_artifacts import iter_tensors
from onnx_graph_taps import sha256


def extract_subgraph(model, inputs, outputs):
    for names in (inputs, outputs):
        if not names or len(set(names)) != len(names) or any(not name for name in names):
            raise ValueError('Nonempty unique boundary names required')
    if set(inputs) & set(outputs):
        raise ValueError('Input and output boundaries must be disjoint')
    if any(t.data_location == onnx.TensorProto.EXTERNAL or t.external_data for t in iter_tensors(model)):
        raise ValueError('Only self-contained graphs supported')
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    info = {v.name: v for v in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)}
    weights = {t.name: t for t in model.graph.initializer}
    for name in (*inputs, *outputs):
        if (name in weights or name not in info or not info[name].type.HasField('tensor_type')
                or not info[name].type.tensor_type.elem_type
                or not info[name].type.tensor_type.HasField('shape')):
            raise ValueError('Known non-initializer tensor dtype/rank required at boundary')
    result = onnx.utils.Extractor(inferred).extract_model(inputs, outputs)
    onnx.checker.check_model(result)
    original_nodes = Counter(n.SerializeToString() for n in model.graph.node)
    if Counter(n.SerializeToString() for n in result.graph.node) - original_nodes:
        raise ValueError('Extraction changed retained node fields')
    if any(t.name not in weights or t.SerializeToString() != weights[t.name].SerializeToString()
           for t in result.graph.initializer):
        raise ValueError('Extraction changed retained weights')
    if not result.graph.node:
        raise ValueError('Nonempty diagnostic subgraph required')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output', action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    source_bytes = args.source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != args.expected_sha256:
        raise ValueError('Source graph hash differs')
    result = extract_subgraph(onnx.load_model_from_string(source_bytes), args.input, args.output)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    graph = args.output_dir / 'diagnostic.onnx'
    onnx.save(result, str(graph))
    onnx.checker.check_model(str(graph))
    receipt = {'schema_version': 1, 'status': 'DIAGNOSTIC_SUBGRAPH_CREATED_UNVALIDATED',
               'source': str(args.source.resolve()), 'source_sha256': source_hash,
               'graph': str(graph.resolve()), 'graph_sha256': sha256(graph),
               'inputs': args.input, 'outputs': args.output,
               'retained_nodes_and_weights_unchanged': True,
               'onnx_version': onnx.__version__, 'tool_sha256': sha256(__file__),
               'warning': 'Extraction can change optimizer/provider partitioning'}
    with (args.output_dir / 'subgraph-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
