"""Opt-in fp32 one-token NeMo TDT head candidate; never automatic promotion.

Preserves the original graph and all non-LSTM operators. The source/config
must be hash-pinned; output is fresh and explicitly unvalidated. This is not
a general recurrent-network converter or an ASR decoder implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from onnx_artifacts import iter_tensors
from export_nemo_fp32 import DECODER_INPUTS, DECODER_OUTPUTS


def digest(data):
    return hashlib.sha256(data).hexdigest()


def clone(model):
    result = onnx.ModelProto()
    result.CopyFrom(model)
    return result


def shape(value):
    tensor = value.type.tensor_type
    if not tensor.HasField('shape'):
        raise ValueError('Known tensor ranks required')
    return tensor.elem_type, list(tensor.shape.dim)


def expand_head(model, config):
    """Pure transformation; no source mutation, downloads, inference or writes."""
    decoder = config['decoder']
    hidden = decoder['prednet']['pred_hidden']
    if (config['loss']['loss_name'] != 'tdt' or decoder['blank_as_pad'] is not True
            or decoder['prednet']['pred_rnn_layers'] != 2 or type(hidden) is not int or hidden <= 0):
        raise ValueError('Only two-layer offline blank-as-pad TDT heads supported')
    if [(v.domain, v.version) for v in model.opset_import] != [('', 17)] or model.functions:
        raise ValueError('Only standard opset 17 without local functions supported')
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in model.graph.node for a in n.attribute):
        raise ValueError('Nested graphs are not supported')
    for tensor in iter_tensors(model):
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError('Self-contained source graph required')
        if tensor.data_type in (TensorProto.FLOAT16, TensorProto.DOUBLE, TensorProto.BFLOAT16):
            raise ValueError('Only fp32 floating tensors supported')
    onnx.checker.check_model(model)
    for values, expected in ((model.graph.input, DECODER_INPUTS), (model.graph.output, DECODER_OUTPUTS)):
        if {v.name for v in values} != set(expected):
            raise ValueError('Family head input/output names differ')
        for value in values:
            dtype, dims = shape(value)
            wanted_dtype, rank = expected[value.name]
            if TensorProto.DataType.Name(dtype) != wanted_dtype or len(dims) != rank:
                raise ValueError('Family head input/output dtype/rank differs')
    inputs = {v.name: v for v in model.graph.input}
    _, target_dims = shape(inputs['targets'])
    if target_dims[1].HasField('dim_value') and target_dims[1].dim_value != 1:
        raise ValueError('Targets sequence axis must admit exactly one token')
    for name in ('input_states_1', 'input_states_2'):
        _, dims = shape(inputs[name])
        if dims[0].dim_value != 2 or dims[2].dim_value != hidden:
            raise ValueError('Config/state width or layer count differs')
    lstms = [n for n in model.graph.node if n.op_type == 'LSTM']
    if len(lstms) != 2:
        raise ValueError('Exactly two LSTMs required')
    result = clone(model)
    next(v for v in result.graph.input if v.name == 'targets').type.tensor_type.shape.dim[1].dim_value = 1
    # Infer afresh: stale value_info must not hide whether targets really bound
    # both recurrent sequence axes. Original metadata is kept in the result.
    probe = clone(result)
    del probe.graph.value_info[:]
    inferred = onnx.shape_inference.infer_shapes(probe, strict_mode=True)
    info = {v.name: v for v in (*inferred.graph.input, *inferred.graph.output, *inferred.graph.value_info)}
    weights = {t.name: t for t in model.graph.initializer}
    node_names = [n.name for n in model.graph.node if n.name]
    if len(node_names) != len(set(node_names)):
        raise ValueError('Duplicate source node names')
    occupied = set(node_names) | set(info) | set(weights)
    for n in model.graph.node:
        occupied.update(n.input)
        occupied.update(n.output)
    nodes, added, removed, contracts = [], [], set(), []

    for node in model.graph.node:
        if node.op_type != 'LSTM':
            nodes.append(node)
            continue
        layer = len(contracts)
        prefix = f'_tdt_step_{layer}/'
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        defaults = {'direction': b'forward', 'layout': 0, 'input_forget': 0,
                    'activations': [b'Sigmoid', b'Tanh', b'Tanh']}
        if (node.domain or attrs.get('hidden_size') != hidden
                or set(attrs) - {'hidden_size', *defaults}
                or any(key in attrs and attrs[key] != value for key, value in defaults.items())):
            raise ValueError('Unsupported LSTM attributes')
        if (len(node.input) != 7 or node.input[4] or not all(node.input[i] for i in (0, 1, 2, 3, 5, 6))
                or len(node.output) != 3 or not all(node.output)):
            raise ValueError('Default LSTM with explicit states and no sequence_lens/P required')
        for name in (node.input[0], node.input[5], node.input[6]):
            if name not in info:
                raise ValueError('Known recurrent tensor shape required')
            dtype, dims = shape(info[name])
            if dtype != TensorProto.FLOAT or len(dims) != 3 or dims[0].dim_value != 1:
                raise ValueError('Recurrent input must be fp32 with proven one-step/direction axis')
        arrays = []
        for name in node.input[1:4]:
            if name not in weights or name in removed:
                raise ValueError('Distinct initializer W/R/B required')
            if any(name in other.input for other in model.graph.node if other is not node):
                raise ValueError('Shared LSTM initializer consumers not supported')
            tensor = weights[name]
            if tensor.data_type != TensorProto.FLOAT:
                raise ValueError('LSTM weights must be fp32')
            value = numpy_helper.to_array(tensor)
            if not np.isfinite(value).all():
                raise ValueError('Finite LSTM weights required')
            arrays.append(value)
        w, r, b = arrays
        input_width = shape(info[node.input[0]])[1][2].dim_value
        if (input_width <= 0 or w.shape != (1, 4 * hidden, input_width)
                or r.shape != (1, 4 * hidden, hidden) or b.shape != (1, 8 * hidden)
                or any(shape(info[node.input[i]])[1][2].dim_value != hidden for i in (5, 6))):
            raise ValueError('LSTM weight/input/state dimensions differ')

        def reserve(suffix):
            name = prefix + suffix
            if name in occupied:
                raise ValueError('Generated name collision')
            occupied.add(name)
            return name

        def constant(suffix, value):
            name = reserve(suffix)
            added.append(numpy_helper.from_array(np.ascontiguousarray(value), name=name))
            return name

        wt = np.ascontiguousarray(w[0].T)
        rt = np.ascontiguousarray(r[0].T)
        wb, rb = b[0, :4 * hidden].copy(), b[0, 4 * hidden:].copy()
        if (wt.T.tobytes() != w[0].tobytes() or rt.T.tobytes() != r[0].tobytes()
                or np.concatenate((wb, rb)).tobytes() != b[0].tobytes()):
            raise ValueError('Weight transformation is not reversible')
        wn, rn = constant('W', wt), constant('R', rt)
        wbn, rbn = constant('Wb', wb), constant('Rb', rb)
        split = constant('split', np.array([hidden] * 4, dtype=np.int64))
        axes = constant('axes', np.array([1], dtype=np.int64))

        def emit(op, feeds, suffixes, outputs=None, **attributes):
            names = outputs if outputs is not None else [reserve(s) for s in suffixes]
            name = reserve('node_' + suffixes[0])
            nodes.append(helper.make_node(op, feeds, names, name=name, **attributes))
            return names

        xw = emit('MatMul', [node.input[0], wn], ['xw'])[0]
        hr = emit('MatMul', [node.input[5], rn], ['hr'])[0]
        linear = emit('Add', [xw, hr], ['linear'])[0]
        bias = emit('Add', [wbn, rbn], ['bias'])[0]
        pre = emit('Add', [linear, bias], ['preactivation'])[0]
        gates = emit('Split', [pre, split], ['i_raw', 'o_raw', 'f_raw', 'g_raw'], axis=2)
        i, o, f = [emit('Sigmoid', [gates[j]], [suffix])[0] for j, suffix in enumerate(('i', 'o', 'f'))]
        g = emit('Tanh', [gates[3]], ['g'])[0]
        fc = emit('Mul', [f, node.input[6]], ['fc'])[0]
        ig = emit('Mul', [i, g], ['ig'])[0]
        emit('Add', [fc, ig], ['cell'], outputs=[node.output[2]])
        ct = emit('Tanh', [node.output[2]], ['cell_tanh'])[0]
        emit('Mul', [o, ct], ['hidden'], outputs=[node.output[1]])
        emit('Unsqueeze', [node.output[1], axes], ['sequence'], outputs=[node.output[0]])
        removed.update(node.input[1:4])
        contracts.append({'source_node': node.name, 'hidden_size': hidden, 'input_width': input_width,
                          'gate_order': 'IOFC', 'weight_bytes_reversible': True})

    del result.graph.node[:]
    result.graph.node.extend(nodes)
    del result.graph.initializer[:]
    result.graph.initializer.extend(t for t in model.graph.initializer if t.name not in removed)
    result.graph.initializer.extend(added)
    onnx.checker.check_model(result)
    onnx.shape_inference.infer_shapes(result, strict_mode=True)
    # Assert exact retained fields and only the intended input-axis constraint.
    before = clone(model)
    after = clone(result)
    for copy in (before, after):
        del copy.graph.node[:]
        del copy.graph.initializer[:]
    next(v for v in before.graph.input if v.name == 'targets').type.tensor_type.shape.dim[1].dim_value = 1
    if before.SerializeToString() != after.SerializeToString():
        raise ValueError('Transformation changed non-target graph metadata')
    return result, contracts


def create_candidate(source, expected_sha256, config_path, expected_config_sha256, output_dir):
    source, config_path, output_dir = Path(source), Path(config_path), Path(output_dir)
    source_bytes, config_bytes = source.read_bytes(), config_path.read_bytes()
    if digest(source_bytes) != expected_sha256 or digest(config_bytes) != expected_config_sha256:
        raise ValueError('Source graph/config hash differs')
    result, contracts = expand_head(onnx.load_model_from_string(source_bytes), json.loads(config_bytes))
    output_dir.mkdir(parents=True, exist_ok=False)
    graph = output_dir / 'decoder_joint-model.onnx'
    graph_bytes = result.SerializeToString()
    with graph.open('xb') as stream:
        stream.write(graph_bytes)
    with (output_dir / 'source-config.json').open('xb') as stream:
        stream.write(config_bytes)
    receipt = {'schema_version': 1, 'status': 'CANDIDATE_GRAPH_CREATED_UNVALIDATED',
               'source': str(source.resolve()), 'source_sha256': digest(source_bytes),
               'source_config_sha256': digest(config_bytes), 'graph_sha256': digest(graph_bytes),
               'graph': str(graph.resolve()), 'tool_sha256': digest(Path(__file__).read_bytes()),
               'onnx_version': onnx.__version__, 'layers': contracts,
               'constraint': 'targets sequence axis 1 fixed to one; batch/encoder-grid axes unchanged',
               'warning': 'No numerical, recurrent decode, dynamic batch/grid, browser or performance acceptance'}
    with (output_dir / 'transformation-receipt.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_candidate(args.source, args.expected_sha256, args.config,
                                     args.expected_config_sha256, args.output_dir)))


if __name__ == '__main__':
    main()
