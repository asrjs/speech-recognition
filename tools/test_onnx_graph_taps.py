"""Small graph instrumentation contracts, not ASR/browser acceptance."""
import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from onnx_graph_taps import add_graph_taps, main, sha256


def graph_fixture():
    graph = helper.make_graph([
        helper.make_node('Neg', ['x'], ['middle']),
        helper.make_node('Relu', ['middle'], ['y'])], 'tap-test',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    return model


class GraphTapTests(unittest.TestCase):
    def test_only_appended_output_and_input_model_immutable(self):
        source = graph_fixture()
        before = source.SerializeToString()
        tapped = add_graph_taps(source, ['middle'])
        self.assertEqual(source.SerializeToString(), before)
        self.assertEqual([v.name for v in tapped.graph.output], ['y', 'middle'])
        del tapped.graph.output[1:]
        self.assertEqual(tapped.SerializeToString(), before)

    def test_real_cpu_ort_existing_output_exact_and_tap_correct(self):
        source = graph_fixture()
        sessions = [ort.InferenceSession(m.SerializeToString(), providers=['CPUExecutionProvider'])
                    for m in (source, add_graph_taps(source, ['middle']))]
        feeds = {'x': np.array([[2, -3]], dtype=np.float32)}
        original = sessions[0].run(None, feeds)
        diagnostic = sessions[1].run(None, feeds)
        np.testing.assert_array_equal(original[0], diagnostic[0])
        np.testing.assert_array_equal(diagnostic[1], -feeds['x'])

    def test_missing_duplicate_empty_existing_output_and_input_rejected(self):
        for names in ([], [''], ['missing'], ['middle', 'middle'], ['y'], ['x']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                add_graph_taps(graph_fixture(), names)

    def test_external_tensor_rejected_without_loading_file(self):
        source = graph_fixture()
        weight = source.graph.initializer.add(name='unused', data_type=TensorProto.FLOAT,
                                              dims=[1], data_location=TensorProto.EXTERNAL)
        weight.external_data.add(key='location', value='missing.bin')
        with self.assertRaisesRegex(ValueError, 'self-contained'):
            add_graph_taps(source, ['middle'])

    def test_invalid_source_checker_failure_does_not_mutate_source(self):
        source = graph_fixture()
        source.graph.node[0].input[0] = 'unbound'
        before = copy.deepcopy(source)
        with self.assertRaises(Exception):
            add_graph_taps(source, ['middle'])
        self.assertEqual(source, before)

    def test_cli_hash_mismatch_and_existing_destination_never_overwrite(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.onnx'
            onnx.save(graph_fixture(), str(source))
            before = source.read_bytes()
            output = Path(directory) / 'existing'
            output.mkdir()
            for digest, error in (('wrong', ValueError), (sha256(source), FileExistsError)):
                with patch('sys.argv', ['onnx_graph_taps.py', '--source', str(source),
                           '--expected-sha256', digest, '--tap', 'middle', '--output-dir', str(output)]):
                    with self.assertRaises(error):
                        main()
                self.assertEqual(source.read_bytes(), before)
                self.assertEqual(list(output.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
