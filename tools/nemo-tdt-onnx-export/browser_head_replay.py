"""Real browser recorded TDT head replay, independent of failing JS features.

Uses the compiled library session bridge, not a public transcribe acceptance run.
No download, install, sibling edits or additional ASR search implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from validate_stage_replay import CONTROL_KEYS, read_json, replay_bindings, stage, log_softmax
from export_nemo_fp32 import audit_graph, DECODER_INPUTS, DECODER_OUTPUTS, sha256
from tensor_parity import compare_tensors

ROOT = Path(__file__).resolve().parents[2]


def tensor(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.ravel().tolist()}


def make_fixture(reference, summary, cfg, identity, audio_hash, *, all_steps=False, step_indices=None):
    if reference["schema_version"] != 1 or len(reference["samples"]) != 1:
        raise ValueError("Single schema-v1 official capture required")
    for item in (reference, summary["stage_capture"]):
        flags = item["instrumentation_control"]
        if set(flags) != CONTROL_KEYS or any(value is not True for value in flags.values()):
            raise ValueError("Official instrumentation controls differ")
    for document in (reference, summary):
        for key, value in identity.items():
            if document["meta"][key] != value:
                raise ValueError("Official source identity differs")
        if document["meta"]["audio_sha256"] != audio_hash:
            raise ValueError("Official audio identity differs")
    sample = reference["samples"][0]
    if (sample["audio"]["sha256"] != audio_hash or reference["meta"]["device"] != "cpu"
            or reference["meta"]["batch_size"] != 1 or reference["stage_order"] != list(sample["stages"])
            or summary["stage_capture"]["stage_count"] != len(sample["stages"])):
        raise ValueError("Official sample/stage contract differs")
    for left, right in (("tokens", "token_ids"), ("transcript", "text"),
                        ("token_frames", "timestamp"), ("token_duration", "token_duration")):
        if sample[left] != summary["result"][right]:
            raise ValueError("Official capture/summary output differs")
    if (cfg["loss"]["loss_name"] != "tdt" or not cfg["decoder"]["blank_as_pad"]
            or cfg["decoder"]["vocab_size"] != cfg["joint"]["num_classes"]
            or cfg["joint"]["log_softmax"] is not None
            or cfg["joint"]["num_extra_outputs"] != len(cfg["decoding"]["durations"])):
        raise ValueError("Unsupported TDT score contract")
    bindings, counts = replay_bindings(sample, cfg)
    if counts != summary["stage_capture"]["call_counts"]:
        raise ValueError("Official call counts differ")
    stages, steps = sample["stages"], []
    channels = cfg["joint"]["num_classes"] + 1 + cfg["joint"]["num_extra_outputs"]
    if step_indices is not None:
        if (all_steps or not step_indices or any(type(i) is not int or i < 0 or i >= len(bindings) for i in step_indices)
                or list(step_indices) != sorted(set(step_indices))):
            raise ValueError("Explicit step indices must be nonempty, unique, increasing and in range; cannot combine with all steps")
        indices = step_indices
    else:
        indices = range(len(bindings)) if all_steps else sorted({0, len(bindings) // 2, len(bindings) - 1})
    for index in indices:
        binding = bindings[index]
        joint, predictor, frame = binding["joint"], binding["predictor"], binding["frame"]
        scores = stage(stages, joint + "/output", "float32", (1, 1, 1, channels))
        if not np.allclose(log_softmax(scores), scores, atol=1e-5, rtol=0):
            raise ValueError("Official joint scores are not all-channel log probabilities")
        steps.append({"binding": binding, "feeds": {
            "encoder_outputs": tensor(np.ascontiguousarray(stage(stages, "encoder", "float32")[:, :, frame:frame+1])),
            "targets": tensor(stage(stages, predictor + "/labels", "int64", (1, 1)).astype(np.int32)),
            "target_length": tensor(np.ones(1, dtype=np.int32)),
            "input_states_1": tensor(stage(stages, predictor + "/input_state_h", "float32")),
            "input_states_2": tensor(stage(stages, predictor + "/input_state_c", "float32"))},
            "expected": {"normalized_scores": tensor(scores),
                "output_states_1": tensor(stage(stages, predictor + "/output_state_h", "float32")),
                "output_states_2": tensor(stage(stages, predictor + "/output_state_c", "float32")),
                "prednet_lengths": tensor(np.ones(1, dtype=np.int32))}})
    return {"sample_id": sample["sample_id"], "blank_id": cfg["decoder"]["vocab_size"], "steps": steps,
            "selection": "explicit-indices" if step_indices is not None else "all-recorded" if all_steps else "first-middle-last",
            "recorded_joint_calls": len(bindings)}


def resolve_route(url, graph):
    route = unquote(urlsplit(url).path)
    if route == "/graph.onnx":
        return graph
    if route == "/replay.mjs":
        return Path(__file__).with_suffix(".mjs")
    for prefix, root, suffixes in (("/dist/", ROOT / "dist", {".js"}),
                                  ("/ort-dist/", ROOT / "node_modules/onnxruntime-web/dist", {".mjs", ".wasm"})):
        if route.startswith(prefix):
            path = (root / route[len(prefix):]).resolve()
            if path.is_relative_to(root.resolve()) and path.suffix in suffixes and path.is_file():
                return path
    raise ValueError("Route is not an allowed executable artifact")


class Handler(BaseHTTPRequestHandler):
    # All successful routes have Content-Length; keep browser asset connections
    # alive instead of the HTTP/1.0 close-delimited lifecycle that reset locally.
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self.event = {"path": self.path, "started_at": time.time(), "bytes_written": 0}
        self.body_hash = hashlib.sha256()
        self.server.http_events.append(self.event)
        try:
            self.serve_get()
            self.event["body_complete"] = self.event.get("status") == 200
        except Exception as error:
            self.event["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            self.event["body_sha256"] = self.body_hash.hexdigest()
            self.event["elapsed_ms"] = (time.time() - self.event["started_at"]) * 1000

    def send_response(self, code, message=None):
        if hasattr(self, "event"):
            self.event["status"] = code
        super().send_response(code, message)

    def write_body(self, data):
        self.wfile.write(data)
        self.event["bytes_written"] += len(data)
        self.body_hash.update(data)

    def serve_get(self):
        if self.path == "/":
            module = "ort.webgpu.min.mjs" if self.server.backend == "webgpu" else "ort.all.bundle.min.mjs"
            data = ('<!doctype html><script type="importmap">' + json.dumps({"imports": {
                "onnxruntime-web": "/ort-dist/" + module}}) + '</script><script type="module" src="/replay.mjs"></script>').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.write_body(data)
            return
        try:
            path = resolve_route(self.path, self.server.graph)
        except ValueError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/javascript" if path.suffix in {".js", ".mjs"} else mimetypes.guess_type(path)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.end_headers()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                self.write_body(block)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *args):
        pass


def evaluate_result(result, fixture, atol=1e-5, rtol=1e-4):
    if result["backendRequested"] not in {"webgpu", "wasm"}:
        raise ValueError("Unknown returned backend")
    if result.get("error") or result["cleanupErrors"]:
        return False
    if len(result["runs"]) != len(fixture["steps"]) * 2:
        raise ValueError("Missing repeated graph calls")
    good = True
    for expected_index, run in enumerate(result["runs"]):
        if (run["repeat"], run["step"]) != divmod(expected_index, len(fixture["steps"])):
            raise ValueError("Repeated call chronology differs")
        values = run["outputs"]
        if set(values) != set(DECODER_OUTPUTS):
            raise ValueError("Actual graph output names differ")
        actual = {name: stage(values, name, "float32" if name != "prednet_lengths" else None) for name in values if name != "prednet_lengths"}
        length = values["prednet_lengths"]
        if length != {"dtype": "int32", "shape": [1], "data": [1]} or type(length["data"][0]) is not int:
            raise ValueError("Predictor length is not exactly int32 one")
        actual["prednet_lengths"] = np.ones(1, dtype=np.int32)
        actual["normalized_scores"] = log_softmax(actual.pop("outputs"))
        expected = fixture["steps"][run["step"]]["expected"]
        run["comparisons"] = {name: compare_tensors(np.asarray(value["data"], dtype=value["dtype"]).reshape(value["shape"]),
                              actual[name], atol=atol, rtol=rtol) for name, value in expected.items()}
        blank = fixture["blank_id"]
        reference = np.asarray(expected["normalized_scores"]["data"], dtype=np.float32)
        scores = actual["normalized_scores"].ravel()
        run["token_duration_argmax_exact"] = all(int(np.argmax(reference[part])) == int(np.argmax(scores[part]))
                                             for part in (slice(0, blank + 1), slice(blank + 1, None)))
        good = all(item["allclose"] for item in run["comparisons"].values()) and run["token_duration_argmax_exact"] and good
        if result["backendRequested"] == "webgpu":
            good = run["gpuDispatches"] > 0 and good
    return good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "summary", "audio", "artifact-dir", "export-receipt", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("wasm", "webgpu"), required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-steps", action="store_true", help="Replay every recorded joint binding, not only first/middle/last")
    selection.add_argument("--step-index", type=int, action="append", help="Zero-based recorded binding index; repeat in increasing order for isolated replay")
    parser.add_argument("--timeout-ms", type=int, default=180000)
    args = parser.parse_args()
    if args.timeout_ms < 1000:
        raise ValueError("Positive bounded timeout required")
    graph = args.artifact_dir.resolve() / "decoder_joint-model.onnx"
    exported = read_json(args.export_receipt)
    artifact = audit_graph(graph, DECODER_INPUTS, DECODER_OUTPUTS)
    if artifact["sha256"] != exported["decoder_joint"]["sha256"] or artifact["external_tensor_count"]:
        raise ValueError("Head export hash differs or external files are not supported")
    packed = read_json(args.artifact_dir / "repack-receipt.json")
    if packed["status"] != "REPACKED_BYTE_IDENTICAL_UNVALIDATED" or packed["source_graph"]["sha256"] != exported["encoder"]["sha256"]:
        raise ValueError("Packed/export source graph identity differs")
    config = args.artifact_dir / "source-config.json"
    expected_config = next(item for item in packed["companions"] if item["name"] == config.name)
    if sha256(config) != expected_config["sha256"] or config.stat().st_size != expected_config["bytes"]:
        raise ValueError("Packed source config identity differs")
    identity = {"model": exported["model_id"], "model_revision": exported["model_revision"], "weight_sha256": exported["source"]["sha256"]}
    fixture = make_fixture(read_json(args.capture), read_json(args.summary), read_json(config), identity, sha256(args.audio), all_steps=args.all_steps, step_indices=args.step_index)
    receipt = {"schema_version": 1, "scope": "RECORDED_HEAD_COMPONENT_ONLY", "status": "IN_PROGRESS",
               "identity": identity, "capture_sha256": sha256(args.capture), "graph": artifact,
               "bridge_sha256": sha256(ROOT / "dist/models/nemo-tdt/ort.js"), "sample_id": fixture["sample_id"],
               "tool_sha256": sha256(Path(__file__)), "browser_module_sha256": sha256(Path(__file__).with_suffix(".mjs")),
               "config_sha256": sha256(config), "summary_sha256": sha256(args.summary), "audio_sha256": sha256(args.audio),
               "export_receipt_sha256": sha256(args.export_receipt),
               "fixture_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True, allow_nan=False).encode()).hexdigest(),
               "bindings": [step["binding"] for step in fixture["steps"]], "atol": 1e-5, "rtol": 1e-4,
               "selection": fixture["selection"], "recorded_joint_calls": fixture["recorded_joint_calls"],
               "features_encoder_sequence_public_api": "UNTESTED", "gpu_scope": "compute dispatch observed, not proof of every node on GPU"}
    with args.output.open("x", encoding="utf-8") as output:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.graph, server.backend = graph, args.backend
        server.http_events = receipt["http_events"] = []
        receipt["server_request_queue_size"] = server.request_queue_size
        receipt["server_protocol"] = Handler.protocol_version
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"])
                try:
                    receipt["browser_version"] = browser.version
                    page = browser.new_page()
                    receipt["browser_errors"] = []
                    page.on("requestfailed", lambda request: receipt["browser_errors"].append({"url": request.url, "failure": request.failure}))
                    page.on("pageerror", lambda error: receipt["browser_errors"].append({"pageerror": str(error)}))
                    page.on("response", lambda response: receipt["browser_errors"].append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
                    page.set_default_timeout(args.timeout_ms)
                    page.goto(f"http://127.0.0.1:{server.server_port}/")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_function("typeof replayRecordedHead === 'function'")
                    # wait_for_function bounds the inference promise; page.evaluate alone has no timeout.
                    page.evaluate("args => { globalThis.pendingReplay = true; replayRecordedHead(...args).then(r => { globalThis.replayResult = r; globalThis.pendingReplay = false; }).catch(e => { globalThis.replayResult = { error: String(e) }; globalThis.pendingReplay = false; }); }", [fixture, args.backend])
                    page.wait_for_function("pendingReplay === false")
                    result = page.evaluate("replayResult")
                    receipt["result"] = result
                    if result["backendRequested"] != args.backend:
                        raise ValueError("Requested/returned backend differs")
                    receipt["status"] = "RECORDED_HEAD_COMPONENT_PASS" if evaluate_result(result, fixture) else "RECORDED_HEAD_COMPONENT_FAIL"
                finally:
                    browser.close()
        except Exception as error:
            receipt["status"], receipt["error"] = "BROWSER_REPLAY_FAILED", str(error)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
    return 0 if receipt["status"] == "RECORDED_HEAD_COMPONENT_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
