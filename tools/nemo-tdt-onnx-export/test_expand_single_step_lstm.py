"""Synthetic graph safety and CPU math, not ASR/browser acceptance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as T, helper as h, numpy_helper as nh

from expand_single_step_lstm import create_candidate, digest, expand_head
from browser_head_replay import validate_candidate


def fixture(hidden=3, input_width=2):
    rng = np.random.default_rng(75)
    weights = []
    nodes = []

    def const(name, value):
        weights.append(nh.from_array(np.asarray(value), name=name))
        return name

    axes = const('axes', np.array([0], dtype=np.int64))
    direction = const('direction', np.array([1], dtype=np.int64))
    nodes.extend([
        h.make_node('Cast', ['targets'], ['labels64'], name='cast', to=T.INT64),
        h.make_node('Gather', [const('embedding', rng.normal(size=(7, input_width)).astype('float32')), 'labels64'], ['embedded'], name='embedding'),
        h.make_node('Transpose', ['embedded'], ['x0'], name='transpose', perm=[1, 0, 2]),
    ])
    for layer in range(2):
        start = const(f'start{layer}', np.array([layer], dtype=np.int64))
        end = const(f'end{layer}', np.array([layer + 1], dtype=np.int64))
        for state, suffix in (('input_states_1', 'h'), ('input_states_2', 'c')):
            nodes.append(h.make_node('Slice', [state, start, end, axes], [f'{suffix}{layer}'], name=f'slice_{suffix}{layer}'))
        names = []
        for suffix, dims in (('W', (1, 4 * hidden, input_width if layer == 0 else hidden)),
                             ('R', (1, 4 * hidden, hidden)), ('B', (1, 8 * hidden))):
            names.append(const(f'{suffix}{layer}', (rng.normal(size=dims) * .25).astype('float32')))
        nodes.append(h.make_node('LSTM', [f'x{layer}', *names, '', f'h{layer}', f'c{layer}'],
                                 [f'y{layer}', f'yh{layer}', f'yc{layer}'], name=f'lstm{layer}', hidden_size=hidden))
        if layer == 0:
            nodes.append(h.make_node('Squeeze', ['y0', direction], ['x1'], name='squeeze'))
    nodes.extend([
        h.make_node('Concat', ['yh0', 'yh1'], ['output_states_1'], name='states_h', axis=0),
        h.make_node('Concat', ['yc0', 'yc1'], ['output_states_2'], name='states_c', axis=0),
        h.make_node('Identity', ['target_length'], ['prednet_lengths'], name='lengths'),
        h.make_node('Identity', ['encoder_outputs'], ['outputs'], name='scores'),
    ])
    inputs = [h.make_tensor_value_info('targets', T.INT32, ['batch', 'sequence']),
              h.make_tensor_value_info('encoder_outputs', T.FLOAT, ['batch', 2, 'grid']),
              h.make_tensor_value_info('target_length', T.INT32, ['batch']),
              *[h.make_tensor_value_info(f'input_states_{i}', T.FLOAT, [2, 'batch', hidden]) for i in (1, 2)]]
    # Family score output is rank 4, not Identity's rank 3.
    nodes[-1] = h.make_node('Unsqueeze', ['encoder_outputs', direction], ['outputs'], name='scores')
    outputs = [h.make_tensor_value_info('outputs', T.FLOAT, ['batch', 1, 2, 'grid']),
               h.make_tensor_value_info('prednet_lengths', T.INT32, ['batch']),
               *[h.make_tensor_value_info(f'output_states_{i}', T.FLOAT, [2, 'batch', hidden]) for i in (1, 2)]]
    model = h.make_model(h.make_graph(nodes, 'tdt-synthetic-only', inputs, outputs, weights),
                        opset_imports=[h.make_opsetid('', 17)], ir_version=8)
    config = {'loss': {'loss_name': 'tdt'}, 'decoder': {'blank_as_pad': True,
              'prednet': {'pred_hidden': hidden, 'pred_rnn_layers': 2}}}
    onnx.checker.check_model(model)
    return model, config


class SingleStepTests(unittest.TestCase):
    def test_source_immutable_retained_fields_and_deterministic_unique_names(self):
        model, config = fixture()
        original = model.SerializeToString()
        result, contracts = expand_head(model, config)
        self.assertEqual(model.SerializeToString(), original)
        self.assertEqual(result.SerializeToString(), expand_head(model, config)[0].SerializeToString())
        self.assertEqual([v['input_width'] for v in contracts], [2, 3])
        self.assertFalse(any(n.op_type == 'LSTM' for n in result.graph.node))
        names = [n.name for n in result.graph.node]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([n.SerializeToString() for n in model.graph.node if n.op_type != 'LSTM'],
                         [n.SerializeToString() for n in result.graph.node if not n.name.startswith('_tdt_step_')])
        before = {t.name: t for t in model.graph.initializer}
        after = {t.name: t for t in result.graph.initializer}
        for name, value in before.items():
            if name[0] not in ('W', 'R', 'B'):
                self.assertEqual(value.SerializeToString(), after[name].SerializeToString())
        for layer in range(2):
            for suffix in ('W', 'R'):
                np.testing.assert_array_equal(nh.to_array(after[f'_tdt_step_{layer}/{suffix}']).T,
                                              nh.to_array(before[f'{suffix}{layer}'])[0])
            bias = np.concatenate([nh.to_array(after[f'_tdt_step_{layer}/{s}']) for s in ('Wb', 'Rb')])
            np.testing.assert_array_equal(bias, nh.to_array(before[f'B{layer}'])[0])
        inputs = {v.name: v for v in result.graph.input}
        self.assertEqual(inputs['targets'].type.tensor_type.shape.dim[1].dim_value, 1)
        self.assertEqual(inputs['targets'].type.tensor_type.shape.dim[0].dim_param, 'batch')
        self.assertEqual(inputs['encoder_outputs'].SerializeToString(), next(v for v in model.graph.input if v.name == 'encoder_outputs').SerializeToString())

    def test_real_cpu_math_with_batch_grid_and_candidate_fed_recurrence(self):
        for hidden, input_width in ((3, 2), (5, 4)):
            with self.subTest(hidden=hidden):
                source, config = fixture(hidden, input_width)
                candidate = expand_head(source, config)[0]
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                sessions = [ort.InferenceSession(m.SerializeToString(), options, providers=['CPUExecutionProvider'])
                            for m in (source, candidate)]
                rng = np.random.default_rng(101)
                for batch, grid in ((1, 1), (2, 4)):
                    states = [(rng.normal(size=(2, batch, hidden)).astype('float32'), rng.normal(size=(2, batch, hidden)).astype('float32'))]
                    states.append(tuple(v.copy() for v in states[0]))
                    for _ in range(7):
                        feeds = {'targets': rng.integers(0, 7, size=(batch, 1), dtype=np.int32),
                                 'target_length': np.ones(batch, dtype=np.int32),
                                 'encoder_outputs': rng.normal(size=(batch, 2, grid)).astype('float32')}
                        runs = []
                        for session, (hidden_state, cell_state) in zip(sessions, states):
                            values = session.run(None, {**feeds, 'input_states_1': hidden_state, 'input_states_2': cell_state})
                            output_names = [value.name for value in session.get_outputs()]
                            runs.append(dict(zip(output_names, values)))
                        for name in runs[0]:
                            np.testing.assert_allclose(runs[0][name], runs[1][name], atol=1e-5, rtol=1e-4)
                        states = [(r['output_states_1'], r['output_states_2']) for r in runs]
                    with self.assertRaises(ort.capi.onnxruntime_pybind11_state.InvalidArgument):
                        sessions[1].run(None, {**feeds, 'targets': np.zeros((batch, 2), dtype=np.int32),
                                             'input_states_1': states[1][0], 'input_states_2': states[1][1]})

    def test_reject_nondefault_attributes(self):
        for key, value in (('direction', 'reverse'), ('layout', 1), ('input_forget', 1),
                           ('clip', 1.), ('activations', ['Relu', 'Tanh', 'Tanh']),
                           ('activation_alpha', [1., 1., 1.])):
            with self.subTest(attribute=key):
                model, cfg = fixture()
                next(n for n in model.graph.node if n.op_type == 'LSTM').attribute.append(h.make_attribute(key, value))
                with self.assertRaises((ValueError, onnx.shape_inference.InferenceError)):
                    expand_head(model, cfg)

    def test_reject_state_weights_sequence_and_config_mismatches(self):
        mutations = [
            lambda m, c: c['decoder']['prednet'].update(pred_hidden=4),
            lambda m, c: c['decoder']['prednet'].update(pred_rnn_layers=1),
            lambda m, c: c['loss'].update(loss_name='rnnt'),
            lambda m, c: setattr(next(v for v in m.graph.input if v.name == 'targets').type.tensor_type.shape.dim[1], 'dim_value', 2),
            lambda m, c: setattr(next(t for t in m.graph.initializer if t.name == 'W0'), 'data_type', T.FLOAT16),
            lambda m, c: next(n for n in m.graph.node if n.op_type == 'LSTM').input.__setitem__(4, 'target_length'),
            lambda m, c: next(n for n in m.graph.node if n.op_type == 'LSTM').input.append('B0'),
            lambda m, c: next(t for t in m.graph.initializer if t.name == 'W0').dims.__setitem__(2, 3),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                model, cfg = fixture()
                mutation(model, cfg)
                with self.assertRaises((ValueError, onnx.checker.ValidationError, onnx.shape_inference.InferenceError)):
                    expand_head(model, cfg)

    def test_reject_external_and_duplicate_or_colliding_names(self):
        for mode in ('external', 'duplicate', 'collision', 'shared'):
            with self.subTest(mode=mode):
                model, cfg = fixture()
                if mode == 'external':
                    model.graph.initializer[0].data_location = T.EXTERNAL
                elif mode == 'duplicate':
                    model.graph.node[1].name = model.graph.node[0].name
                elif mode == 'collision':
                    model.graph.node[0].name = '_tdt_step_0/node_xw'
                else:
                    model.graph.node.append(h.make_node('Identity', ['W0'], ['shared_weight'], name='shared'))
                with self.assertRaises(ValueError):
                    expand_head(model, cfg)

    def test_refuses_unproven_recurrent_sequence_axis(self):
        model, cfg = fixture()
        # Swap embedding transpose so the LSTM sequence derives from dynamic
        # batch, not targets' constrained token axis. Do not assume shape parity.
        next(n for n in model.graph.node if n.name == 'transpose').attribute[0].ints[:] = [0, 1, 2]
        with self.assertRaises(ValueError):
            expand_head(model, cfg)

    def test_hash_pinning_fresh_directory_and_receipt(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            model, cfg = fixture()
            graph, config = directory / 'source.onnx', directory / 'config.json'
            graph.write_bytes(model.SerializeToString())
            config.write_text(json.dumps(cfg), encoding='utf-8')
            graph_hash, config_hash = digest(graph.read_bytes()), digest(config.read_bytes())
            dest = directory / 'candidate'
            for hashes in (('bad', config_hash), (graph_hash, 'bad')):
                with self.assertRaises(ValueError):
                    create_candidate(graph, hashes[0], config, hashes[1], dest)
                self.assertFalse(dest.exists())
            receipt = create_candidate(graph, graph_hash, config, config_hash, dest)
            self.assertEqual(receipt['status'], 'CANDIDATE_GRAPH_CREATED_UNVALIDATED')
            self.assertEqual(receipt['graph_sha256'], digest((dest / 'decoder_joint-model.onnx').read_bytes()))
            self.assertEqual((dest / 'source-config.json').read_bytes(), config.read_bytes())
            with self.assertRaises(FileExistsError):
                create_candidate(graph, graph_hash, config, config_hash, dest)
            self.assertEqual(digest(graph.read_bytes()), graph_hash)
            self.assertEqual(digest(config.read_bytes()), config_hash)

    def test_hash_pinned_buffer_is_the_buffer_parsed_and_empty_destination_refused(self):
        with TemporaryDirectory() as root:
            directory = Path(root)
            model, cfg = fixture()
            graph, config = directory / 'source.onnx', directory / 'config.json'
            source_bytes = model.SerializeToString()
            graph.write_bytes(source_bytes)
            config.write_text(json.dumps(cfg), encoding='utf-8')
            dest = directory / 'candidate'
            with patch('expand_single_step_lstm.onnx.load_model_from_string', wraps=onnx.load_model_from_string) as loader:
                create_candidate(graph, digest(source_bytes), config, digest(config.read_bytes()), dest)
                self.assertEqual(loader.call_args.args[0], source_bytes)
            empty = directory / 'empty'
            empty.mkdir()
            with self.assertRaises(FileExistsError):
                create_candidate(graph, digest(source_bytes), config, digest(config.read_bytes()), empty)
            self.assertEqual(list(empty.iterdir()), [])

    def test_browser_candidate_gate_reproduces_graph_and_rejects_forged_receipts(self):
        for mode in ('valid', 'tool', 'config', 'layers', 'graph'):
            with self.subTest(mode=mode), TemporaryDirectory() as root:
                directory = Path(root)
                model, cfg = fixture()
                graph, config = directory / 'source.onnx', directory / 'config.json'
                graph.write_bytes(model.SerializeToString())
                config.write_text(json.dumps(cfg), encoding='utf-8')
                graph_hash, config_hash = digest(graph.read_bytes()), digest(config.read_bytes())
                dest = directory / 'candidate'
                receipt = create_candidate(graph, graph_hash, config, config_hash, dest)
                if mode == 'tool':
                    receipt['tool_sha256'] = 'forged'
                elif mode == 'config':
                    (dest / 'source-config.json').write_text('{}', encoding='utf-8')
                elif mode == 'layers':
                    receipt['layers'][0]['hidden_size'] = 100
                elif mode == 'graph':
                    altered = onnx.load(dest / 'decoder_joint-model.onnx')
                    altered.graph.doc_string = 'valid but not the reproduced graph'
                    altered_bytes = altered.SerializeToString()
                    (dest / 'decoder_joint-model.onnx').write_bytes(altered_bytes)
                    receipt['graph_sha256'] = digest(altered_bytes)
                (dest / 'transformation-receipt.json').write_text(json.dumps(receipt), encoding='utf-8')
                if mode == 'valid':
                    candidate, artifact, control = validate_candidate(graph, config, dest, graph_hash, config_hash)
                    self.assertEqual(candidate, dest / 'decoder_joint-model.onnx')
                    self.assertEqual(artifact['sha256'], receipt['graph_sha256'])
                    self.assertTrue(control['reproduced_graph_hash_exact'])
                else:
                    with self.assertRaises(ValueError):
                        validate_candidate(graph, config, dest, graph_hash, config_hash)


if __name__ == '__main__':
    unittest.main()
