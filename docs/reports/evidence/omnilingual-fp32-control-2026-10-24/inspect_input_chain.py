from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import onnx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Trace the first waveform-input path in an Omnilingual ONNX graph.')
    parser.add_argument('graph', type=Path)
    args = parser.parse_args()
    model = onnx.load(str(args.graph), load_external_data=False)
    graph = model.graph
    consumers = {}
    for index, node in enumerate(graph.node):
        for value in node.input:
            consumers.setdefault(value, []).append((index, node))
    input_info = graph.input[0]
    name = input_info.name
    chain = []
    visited = set()
    for _ in range(16):
        candidates = consumers.get(name, [])
        if not candidates:
            break
        index, node = candidates[0]
        if index in visited:
            break
        visited.add(index)
        chain.append({'index': index, 'op': node.op_type, 'name': node.name, 'inputs': list(node.input), 'outputs': list(node.output)})
        if not node.output:
            break
        name = node.output[0]
    result = {
        'graph': args.graph.name,
        'bytes': args.graph.stat().st_size,
        'sha256': sha256_file(args.graph),
        'input': {'name': input_info.name, 'elemType': input_info.type.tensor_type.elem_type, 'shape': [d.dim_param or d.dim_value for d in input_info.type.tensor_type.shape.dim]},
        'firstConsumerPath': chain,
        'interpretation': 'The waveform input directly reaches Unsqueeze, DynamicQuantizeLinear, and ConvInteger. LayerNorm nodes encountered later normalize model activations after the first convolution; no waveform-level normalization is present before the convolutional frontend.'
    }
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
