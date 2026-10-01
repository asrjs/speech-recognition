"""Weight-free prerequisite/receipt tests; not real browser acceptance."""
import copy
import hashlib
import http.client
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
import urllib.request

import numpy as np

from browser_head_replay import Handler, ThreadingHTTPServer, make_fixture, evaluate_result, resolve_route, tensor
from test_validate_stage_replay import sample_fixture
from validate_stage_replay import CONTROL_KEYS, log_softmax


def fixture_documents():
    sample, cfg = sample_fixture()
    cfg.update(loss={"loss_name": "tdt"}, decoding={"durations": [0, 1]})
    cfg["decoder"]["blank_as_pad"] = True
    cfg["joint"].update(num_classes=2, num_extra_outputs=2, log_softmax=None)
    for index in range(2):
        sample["stages"][f"joint/{index:04d}/output"] = tensor(log_softmax(np.array([[[[2, 1, 0, 3, 4]]]], dtype=np.float32)))
    sample.update(sample_id="synthetic", audio={"sha256": "audio"}, tokens=[1], transcript="x", token_frames=[0], token_duration=[1])
    identity = {"model": "test", "model_revision": "rev", "weight_sha256": "weights"}
    meta = {**identity, "audio_sha256": "audio", "device": "cpu", "batch_size": 1}
    controls = dict.fromkeys(CONTROL_KEYS, True)
    reference = {"schema_version": 1, "samples": [sample], "meta": meta, "instrumentation_control": controls,
                 "stage_order": list(sample["stages"])}
    summary = {"meta": meta.copy(), "result": {"token_ids": [1], "text": "x", "timestamp": [0], "token_duration": [1]},
               "stage_capture": {"instrumentation_control": controls.copy(), "stage_count": len(sample["stages"]),
                                 "call_counts": {"encoder_projection": 1, "predictor": 1, "predictor_projection": 1, "joint": 2}}}
    return reference, summary, cfg, identity


def result_fixture(fixture, backend="wasm"):
    runs = []
    for repeat in range(2):
        for index, step in enumerate(fixture["steps"]):
            outputs = copy.deepcopy(step["expected"])
            outputs["outputs"] = outputs.pop("normalized_scores")
            runs.append({"repeat": repeat, "step": index, "outputs": outputs, "gpuDispatches": 1})
    return {"backendRequested": backend, "cleanupErrors": [], "runs": runs}


class BrowserHeadTests(unittest.TestCase):
    def setUp(self):
        self.documents = fixture_documents()
        self.fixture = make_fixture(*self.documents, "audio")

    def test_mapping_uses_raw_encoder_not_projected_joint_input(self):
        self.assertEqual([s["binding"]["frame"] for s in self.fixture["steps"]], [0, 1])
        self.assertEqual(self.fixture["steps"][1]["feeds"]["encoder_outputs"],
                         {"dtype": "float32", "shape": [1, 2, 1], "data": [2, 4]})
        self.assertEqual(self.fixture["steps"][0]["feeds"]["targets"]["dtype"], "int32")

    def test_all_steps_preserves_every_binding_and_rejects_missing_call(self):
        reference, summary, cfg, identity = copy.deepcopy(self.documents)
        sample = reference["samples"][0]
        for index in (2, 3):
            for suffix in ("input_f", "input_g", "output"):
                sample["stages"][f"joint/{index:04d}/{suffix}"] = copy.deepcopy(sample["stages"][f"joint/0001/{suffix}"])
            sample["calls"].append({"kind": "joint", "index": index, "stage_prefix": f"joint/{index:04d}"})
        reference["stage_order"] = list(sample["stages"])
        summary["stage_capture"].update(stage_count=len(sample["stages"]))
        summary["stage_capture"]["call_counts"]["joint"] = 4
        selected = make_fixture(reference, summary, cfg, identity, "audio")
        complete = make_fixture(reference, summary, cfg, identity, "audio", all_steps=True)
        self.assertEqual(selected["selection"], "first-middle-last")
        self.assertEqual([s["binding"]["joint"] for s in selected["steps"]], ["joint/0000", "joint/0002", "joint/0003"])
        self.assertEqual(complete["selection"], "all-recorded")
        self.assertEqual(complete["recorded_joint_calls"], 4)
        self.assertEqual([s["binding"]["joint"] for s in complete["steps"]], [f"joint/{i:04d}" for i in range(4)])
        self.assertEqual([s["binding"]["predictor"] for s in complete["steps"]], ["predictor/0000"] * 4)
        self.assertTrue(evaluate_result(result_fixture(complete), complete))
        result = result_fixture(complete)
        result["runs"].pop(1)
        with self.assertRaisesRegex(ValueError, "Missing repeated graph calls"):
            evaluate_result(result, complete)

    def test_identity_control_and_summary_fail_before_browser(self):
        for mutate in (lambda d: d[0]["meta"].update(model_revision="other"),
                       lambda d: d[1]["stage_capture"]["instrumentation_control"].update(after_text_match=False),
                       lambda d: d[1]["result"].update(text="other"),
                       lambda d: d[0]["samples"][0]["audio"].update(sha256="other")):
            documents = copy.deepcopy(self.documents)
            mutate(documents)
            with self.assertRaises(ValueError):
                make_fixture(*documents, "audio")

    def test_raw_or_nonfinite_reference_scores_fail(self):
        for values in ([2, 1, 0, 3, 4], [float("nan")] * 5):
            documents = copy.deepcopy(self.documents)
            documents[0]["samples"][0]["stages"]["joint/0000/output"]["data"] = values
            with self.assertRaises(ValueError):
                make_fixture(*documents, "audio")

    def test_full_output_comparison_and_repeat_chronology(self):
        result = result_fixture(self.fixture)
        self.assertTrue(evaluate_result(result, self.fixture))
        self.assertTrue(all(run["token_duration_argmax_exact"] for run in result["runs"]))
        result = result_fixture(self.fixture)
        result["runs"][1]["step"] = 0
        with self.assertRaises(ValueError):
            evaluate_result(result, self.fixture)

    def test_missing_state_wrong_shape_nonfinite_and_length_fail(self):
        for mutate in (lambda o: o.pop("output_states_1"),
                       lambda o: o["output_states_1"].update(shape=[1, 2]),
                       lambda o: o["outputs"]["data"].__setitem__(4, float("nan")),
                       lambda o: o["prednet_lengths"].update(dtype="int64")):
            result = result_fixture(self.fixture)
            mutate(result["runs"][0]["outputs"])
            with self.assertRaises(ValueError):
                evaluate_result(result, self.fixture)

    def test_numerical_failure_cleanup_and_gpu_zero_dispatch_are_not_pass(self):
        result = result_fixture(self.fixture)
        result["runs"][0]["outputs"]["output_states_1"]["data"][0] = 1
        self.assertFalse(evaluate_result(result, self.fixture))
        result = result_fixture(self.fixture)
        result["cleanupErrors"] = ["release failed"]
        self.assertFalse(evaluate_result(result, self.fixture))
        result = result_fixture(self.fixture, "webgpu")
        result["runs"][0]["gpuDispatches"] = 0
        self.assertFalse(evaluate_result(result, self.fixture))

    def test_server_rejects_traversal_and_nonexecutable_files(self):
        graph = Path("graph.onnx")
        self.assertEqual(resolve_route("/graph.onnx", graph), graph)
        for url in ("/dist/../../package.json", "/dist/%2e%2e/package.json", "/ort-dist/../package.json", "/source-config.json"):
            with self.assertRaises(ValueError):
                resolve_route(url, graph)

    def test_server_records_complete_response_bytes_hash_and_missing_route(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.onnx"
            payload = b"synthetic graph delivery, not inference"
            path.write_bytes(payload)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.graph, server.backend, server.http_events = path, "wasm", []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(base + "/graph.onnx", timeout=5) as response:
                    self.assertEqual(response.version, 11)
                    self.assertEqual(response.read(), payload)
                    self.assertEqual(response.headers["Cross-Origin-Embedder-Policy"], "require-corp")
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(base + "/not-allowed", timeout=5)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            complete, missing = server.http_events
            self.assertTrue(complete["body_complete"])
            self.assertEqual(complete["bytes_written"], len(payload))
            self.assertEqual(complete["body_sha256"], hashlib.sha256(payload).hexdigest())
            self.assertEqual(missing["status"], 404)
            self.assertFalse(missing["body_complete"])

    def test_http11_reuses_connection_with_exact_response_framing(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "graph.onnx"
            payload = b"two responses on one connection"
            path.write_bytes(payload)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            server.graph, server.backend, server.http_events = path, "wasm", []
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            try:
                connection.request("GET", "/graph.onnx")
                response = connection.getresponse()
                self.assertEqual(response.version, 11)
                self.assertEqual(response.read(), payload)
                socket = connection.sock
                self.assertIsNotNone(socket)
                connection.request("GET", "/graph.onnx")
                self.assertEqual(connection.getresponse().read(), payload)
                self.assertIs(connection.sock, socket)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
            self.assertEqual(len(server.http_events), 2)
            self.assertTrue(all(event["body_complete"] for event in server.http_events))


if __name__ == "__main__":
    unittest.main()
