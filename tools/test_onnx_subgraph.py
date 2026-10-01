"""Diagnostic boundary controls, not production/browser acceptance."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

from onnx_subgraph import extract_subgraph, main, sha256


def fixture():
    graph = helper.make_graph([
        helper.make_node('Neg', ['x'], ['middle']),
        helper.make_node('Add', ['middle', 'bias'], ['y'])], 'subgraph-test',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor('bias', TensorProto.FLOAT, [1, 2], [2, 3])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    return model


class SubgraphTests(unittest.TestCase):
    def test_original_immutable_and_retained_node_weight_exact(self):
        source = fixture()
        before = source.SerializeToString()
        sub = extract_subgraph(source, ['middle'], ['y'])
        self.assertEqual(source.SerializeToString(), before)
        self.assertEqual(len(sub.graph.node), 1)
        self.assertEqual(sub.graph.node[0], source.graph.node[1])
        self.assertEqual(sub.graph.initializer[0], source.graph.initializer[0])
        self.assertEqual([v.name for v in sub.graph.input], ['middle'])

    def test_real_cpu_matched_boundary_exact(self):
        source = fixture()
        sub = extract_subgraph(source, ['middle'], ['y'])
        full = ort.InferenceSession(source.SerializeToString(), providers=['CPUExecutionProvider'])
        part = ort.InferenceSession(sub.SerializeToString(), providers=['CPUExecutionProvider'])
        x = np.array([[4, -5]], dtype=np.float32)
        np.testing.assert_array_equal(full.run(None, {'x': x})[0], part.run(None, {'middle': -x})[0])

    def test_invalid_boundary_names_rejected(self):
        for inputs, outputs in (([], ['y']), (['middle'], []), ([''], ['y']),
                                (['middle', 'middle'], ['y']), (['middle'], ['y', 'y']),
                                (['middle'], ['middle']), (['missing'], ['y']), (['bias'], ['y'])):
            with self.subTest(inputs=inputs, outputs=outputs), self.assertRaises(ValueError):
                extract_subgraph(fixture(), inputs, outputs)

    def test_external_data_rejected_without_loading(self):
        source = fixture()
        weight = source.graph.initializer[0]
        weight.ClearField('float_data')
        weight.data_location = TensorProto.EXTERNAL
        weight.external_data.add(key='location', value='missing.bin')
        with self.assertRaisesRegex(ValueError, 'self-contained'):
            extract_subgraph(source, ['middle'], ['y'])

    def test_incomplete_cut_rejected_by_checker(self):
        with self.assertRaises(onnx.checker.ValidationError):
            extract_subgraph(fixture(), ['y'], ['middle'])

    def test_cli_success_hash_and_destination_safety(self):
        with TemporaryDirectory() as directory:
            source = Path(directory) / 'source.onnx'
            onnx.save(fixture(), str(source))
            before = source.read_bytes()
            output = Path(directory) / 'output'
            argv = ['onnx_subgraph.py', '--source', str(source), '--expected-sha256', sha256(source),
                    '--input', 'middle', '--output', 'y', '--output-dir', str(output)]
            with patch('sys.argv', argv), patch('onnx_subgraph.onnx.load_model_from_string',
                                               wraps=onnx.load_model_from_string) as load:
                main()
                load.assert_called_once_with(before)
            receipt = json.loads((output / 'subgraph-receipt.json').read_text())
            self.assertEqual(receipt['graph_sha256'], sha256(output / 'diagnostic.onnx'))
            self.assertEqual(receipt['status'], 'DIAGNOSTIC_SUBGRAPH_CREATED_UNVALIDATED')
            for digest, error in (('wrong', ValueError), (sha256(source), FileExistsError)):
                argv[4] = digest
                with patch('sys.argv', argv), self.assertRaises(error):
                    main()
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(receipt, json.loads((output / 'subgraph-receipt.json').read_text()))


if __name__ == '__main__':
    unittest.main()
