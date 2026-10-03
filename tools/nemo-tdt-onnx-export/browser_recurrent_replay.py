"""Conditional public-library recurrence with frozen official encoder output.

No feature/real-encoder acceptance, no alternate decoder, no downloads or sibling
edits. The library feeds its own recurrent states to a reproducible real head.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import threading
from pathlib import Path

import numpy as np
import onnx

from browser_head_replay import ROOT, Handler, ThreadingHTTPServer, make_fixture, validate_candidate, resolve_route
from export_nemo_fp32 import sha256
from live_head_parity import compare_live_head
from frozen_encoder_replay import make_graph, verify_delivery, SCOPE
from validate_stage_replay import read_json, stage
from tensor_parity import compare_tensors


def pcm_fixture(sample, rate):
    pcm = stage(sample["stages"], "waveform", "float32")
    if pcm.ndim != 2 or pcm.shape[0] != 1:
        raise ValueError("Batch-one mono PCM required")
    original = sample["stages"]["waveform"]["data"]
    if any(type(value) not in (int, float) or np.float32(value) != value for value in original):
        raise ValueError("PCM values must be exactly f32, not silently rounded")
    length = sample["stages"]["waveform_length"]
    if (length["dtype"] not in {"int32", "int64"} or length["shape"] != [1]
            or length["data"] != [pcm.shape[1]] or type(length["data"][0]) is not int):
        raise ValueError("PCM valid length differs")
    digest = hashlib.sha256(pcm.astype("<f4").tobytes()).hexdigest()
    expected = {"boundary": "preprocessor_input_signal", "sample_rate": rate, "channels": 1,
                "dtype": "float32", "encoding": "f32le", "shape": list(pcm.shape),
                "samples": pcm.shape[1], "valid_samples": pcm.shape[1], "right_padding_samples": 0,
                "sha256": digest}
    if any(sample["model_input"].get(key) != value for key, value in expected.items()):
        raise ValueError("Captured PCM identity/layout/rate differs")
    # The installed Playwright numeric serializer maps Python +0.0 to JS -0.
    # Transport f32le bytes, not numeric lists; preserve both signed zero bits.
    return {"pcmF32leBase64": base64.b64encode(pcm.astype("<f4").tobytes()).decode("ascii"),
            "pcmSha256": digest, "sampleRate": rate}


def final_state_fixture(sample, head, cfg):
    """Match the library's committed cache before consuming its last token.

    Select the predictor output associated with the last NONBLANK JOINT, not
    NeMo's later post-last-label retained predictor state. With no tokens the
    library exposes zero pre-SOS cache, not NeMo's retained SOS output. These
    are deliberately different boundaries. No frame-advance/search logic here.
    """
    if cfg["decoding"]["durations"] != list(range(cfg["joint"]["num_extra_outputs"])):
        raise ValueError("Library sequential duration-index contract differs")
    emitted, frames, durations, accepted = [], [], [], None
    for step in head["steps"]:
        scores = np.asarray(step["expected"]["normalized_scores"]["data"], dtype=np.float32)
        token = int(np.argmax(scores[:head["blank_id"] + 1]))
        if token != head["blank_id"]:
            emitted.append(token)
            frames.append(step["binding"]["frame"])
            durations.append(int(np.argmax(scores[head["blank_id"] + 1:])))
            accepted = step
    if (emitted != sample["tokens"] or frames != sample["token_frames"] or durations != sample["token_duration"]):
        raise ValueError("Recorded nonblank decisions differ from official output")
    first = head["steps"][0]
    if (first["feeds"]["targets"]["data"] != [head["blank_id"]]
            or any(value != 0 for key in ("input_states_1", "input_states_2") for value in first["feeds"][key]["data"])):
        raise ValueError("Fresh blank/zero predictor initial state required")
    return {key: (accepted["expected"][f"output_states_{number}"] if accepted
                  else first["feeds"][f"input_states_{number}"])
            for key, number in (("h", 1), ("c", 2))}


def evaluate_result(result, expected, backend):
    encoder_backend = expected.get('encoderBackend', 'wasm')
    if (result["backendRequested"] != backend or result["encoderBackend"] != encoder_backend
            or result["decoderBackend"] != backend or result.get("loaded") != {"backend": backend, "family": "nemo-tdt"}
            or result.get("pcmSha256") != expected["pcmSha256"] or not result["crossOriginIsolated"]):
        raise ValueError("Actual backend/family/PCM/isolation differs")
    if result.get("error") or result["cleanupErrors"]:
        return False
    if len(result["runs"]) != 2:
        raise ValueError("Exactly two same-handle transcriptions required")
    good = True
    for repeat, run in enumerate(result["runs"]):
        if type(run["repeat"]) is not int or run["repeat"] != repeat:
            raise ValueError("Repeat chronology differs")
        if expected.get('realEncoder') is True:
            metrics = run.get('metrics', {})
            if (result.get('realEncoder') != 'EXECUTED_NOT_COMPONENT_ACCEPTED'
                    or metrics.get('encoderBackend') != encoder_backend
                    or metrics.get('decoderBackend') != backend):
                raise ValueError('Real encoder component execution evidence differs')
        run["output_checks"] = {}
        for key in ("tokenIds", "frameIndices", "tdtSteps"):
            value = run[key]
            if not isinstance(value, list) or any(type(item) is not int for item in value):
                raise ValueError("Explicit integer output lists required")
            run["output_checks"][key] = value == expected[key]
        if not isinstance(run["text"], str):
            raise ValueError("Explicit canonical text required")
        run["output_checks"]["text"] = run["text"] == expected["text"]
        if not run["decoderState"] or set(run["decoderState"]) != {"h", "c"}:
            raise ValueError("Final decoder state required")
        run["state_comparisons"] = {key: compare_tensors(
            np.asarray(value["data"], dtype=np.float32).reshape(value["shape"]),
            stage(run["decoderState"], key, "float32", value["shape"]), atol=1e-5, rtol=1e-4)
            for key, value in expected["decoderState"].items()}
        good = all(run["output_checks"].values()) and all(item["allclose"] for item in run["state_comparisons"].values()) and good
        if backend == "webgpu" or encoder_backend == 'webgpu':
            good = type(run["gpuDispatches"]) is int and run["gpuDispatches"] > 0 and good
    return good


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "summary", "audio", "artifact-dir", "export-receipt", "nemo",
                 "frozen-dir", "candidate-dir", "runtime-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--backend", choices=("wasm", "webgpu"), required=True)
    parser.add_argument('--real-encoder-backend', choices=('wasm', 'webgpu'),
                        help='Opt-in real encoder output diagnostic; never full feature/ASR acceptance')
    parser.add_argument("--trace-head", action="store_true")
    parser.add_argument("--trace-encoder", action="store_true",
                        help="Opt-in: record the encoder's actual inputs and outputs in the same run as the head trace")
    parser.add_argument("--timeout-ms", type=int, default=180000)
    args = parser.parse_args()
    if args.timeout_ms < 1000 or args.output.exists():
        raise ValueError("Fresh output and positive bounded timeout required")
    exported, packed = read_json(args.export_receipt), read_json(args.artifact_dir / "repack-receipt.json")
    source_graph, config = args.artifact_dir / "encoder-model.onnx", args.artifact_dir / "source-config.json"
    cfg, runtime_config = read_json(config), read_json(args.runtime_config)
    config_binding = next(item for item in packed["companions"] if item["name"] == config.name)
    if (packed["status"] != "REPACKED_BYTE_IDENTICAL_UNVALIDATED"
            or packed["source_graph"]["sha256"] != exported["encoder"]["sha256"]
            or sha256(source_graph) != packed["graph"]["sha256"]
            or sha256(config) != config_binding["sha256"] or sha256(args.nemo) != exported["source"]["sha256"]):
        raise ValueError("Source lineage differs")
    for key, value in {"sampleRate": cfg["preprocessor"]["sample_rate"], "frameShiftSeconds": cfg["preprocessor"]["window_stride"],
                       "subsamplingFactor": cfg["encoder"]["subsampling_factor"], "melBins": cfg["preprocessor"]["features"],
                       "vocabularySize": cfg["decoder"]["vocab_size"] + 1, "predictionHiddenSize": cfg["decoder"]["prednet"]["pred_hidden"],
                       "predictionLayers": cfg["decoder"]["prednet"]["pred_rnn_layers"],
                       "maxSymbolsPerStep": cfg["decoding"]["greedy"]["max_symbols"]}.items():
        if runtime_config.get(key) != value:
            raise ValueError(f"Runtime/source config differs: {key}")
    graph, artifact, candidate = validate_candidate(args.artifact_dir / "decoder_joint-model.onnx", config, args.candidate_dir,
                                                    exported["decoder_joint"]["sha256"], config_binding["sha256"])
    reference, summary = read_json(args.capture), read_json(args.summary)
    identity = {"model": exported["model_id"], "model_revision": exported["model_revision"], "weight_sha256": exported["source"]["sha256"]}
    head = make_fixture(reference, summary, cfg, identity, sha256(args.audio), all_steps=True)
    sample = reference["samples"][0]
    frozen_graph = args.frozen_dir / "diagnostic-frozen-encoder.onnx"
    frozen_receipt = read_json(args.frozen_dir / "frozen-boundary-receipt.json")
    model, feeds, outputs = make_graph(onnx.load(str(source_graph), load_external_data=False), sample["stages"], cfg, sample["sample_id"])
    payload = model.SerializeToString()
    if (frozen_receipt["scope"] != SCOPE or frozen_receipt["status"] != "FROZEN_BOUNDARY_NATIVE_DELIVERY_PASS"
            or frozen_receipt["graph_sha256"] != sha256(frozen_graph) or frozen_graph.read_bytes() != payload
            or frozen_receipt["capture_sha256"] != sha256(args.capture) or frozen_receipt["identity"] != identity):
        raise ValueError("Frozen graph/capture identity differs from reproduced boundary")
    native_control = verify_delivery(payload, feeds, outputs)
    tokenizer = args.artifact_dir / "vocab.txt"
    token_binding = next(item for item in packed["companions"] if item["name"] == tokenizer.name)
    if sha256(tokenizer) != token_binding["sha256"]:
        raise ValueError("Tokenizer identity differs")
    fixture = {**pcm_fixture(sample, runtime_config["sampleRate"]), "modelId": identity["model"], "config": runtime_config}
    if args.trace_head and not 0 < len(head["steps"]) <= 2000:
        raise ValueError("Live head trace requires 1..2000 official calls")
    fixture.update(traceHead=args.trace_head, maxTraceCalls=len(head["steps"]),
                   traceEncoder=args.trace_encoder,
                   realEncoder=args.real_encoder_backend is not None,
                   encoderBackend=args.real_encoder_backend or 'wasm')
    expected = {**fixture, "tokenIds": sample["tokens"], "frameIndices": sample["token_frames"],
                "tdtSteps": sample["token_duration"], "text": sample["transcript"], "decoderState": final_state_fixture(sample, head, cfg)}
    module = Path(__file__).with_suffix(".mjs")
    routes = {"/recurrent.mjs": module, "/frozen-encoder.onnx": frozen_graph, "/vocab.txt": tokenizer,
              "/observe-session.mjs": Path(__file__).with_name("observe_session.mjs"),
              "/vendor/pako.esm.mjs": ROOT / "node_modules/pako/dist/pako.esm.mjs"}
    if args.real_encoder_backend:
        data_binding = packed['external_data']
        data = args.artifact_dir / 'encoder-model.onnx.data'
        if (data_binding['name'] != data.name or data.is_symlink() or not data.is_file()
                or data.stat().st_size != data_binding['bytes'] or sha256(data) != data_binding['sha256']):
            raise ValueError('Real encoder external-data identity differs')
        routes.update({'/encoder-model.onnx': source_graph, '/encoder-model.onnx.data': data})
    bound_paths = {*routes.values(), graph, source_graph, config, Path(__file__),
                   Path(__file__).with_name("live_head_parity.py"),
                   args.artifact_dir / "repack-receipt.json", args.frozen_dir / "frozen-boundary-receipt.json",
                   args.candidate_dir / "transformation-receipt.json",
                   *[Path(__file__).with_name(name) for name in (
                       "browser_head_replay.py", "frozen_encoder_replay.py", "validate_stage_replay.py",
                       "export_nemo_fp32.py", "expand_single_step_lstm.py")],
                   ROOT / "tools/tensor_parity.py", ROOT / "tools/onnx_artifacts.py",
                   *list((ROOT / "dist").rglob("*.js")),
                   *[path for path in (ROOT / "node_modules/onnxruntime-web/dist").iterdir()
                     if path.suffix in {".mjs", ".wasm"}]}
    pinned = {str(path.resolve()): sha256(path) for path in bound_paths}
    inputs = {name: {"path": str(path.resolve()), "sha256": sha256(path)} for name, path in vars(args).items() if isinstance(path, Path) and path.is_file()}
    receipt = {"schema_version": 1, "scope": "CONDITIONAL_FROZEN_ENCODER_LIBRARY_RECURRENCE", "status": "IN_PROGRESS",
               "identity": identity, "inputs": inputs, "pinned_artifacts": pinned,
               "candidate": candidate, "decoder_graph": artifact,
               "frozen_graph_sha256": sha256(frozen_graph), "native_frozen_control": native_control,
               "tool_sha256": sha256(Path(__file__)), "browser_module_sha256": sha256(module),
               "features": "EXECUTED_NOT_ACCEPTED", "real_encoder": "BYPASSED_FROZEN_ORACLE", "full_asr": "NOT_PROVEN",
               "state_boundary": "LIBRARY_COMMITTED_PREDICTOR_BEFORE_LAST_EMITTED_TOKEN_OR_PRE_SOS_ZERO",
               "official_post_last_label_retained_state": "DIFFERENT_BOUNDARY_NOT_COMPARED",
               "intermediate_states_logits": "UNTESTED", "atol": 1e-5, "rtol": 1e-4,
               "gpu_scope": "head compute dispatch observed, not every-node GPU placement or VRAM measurement",
               "browser_errors": [], "http_events": []}
    success_status = 'CONDITIONAL_RECURRENCE_PASS'
    if args.real_encoder_backend:
        success_status = 'REAL_ENCODER_OUTPUT_DIAGNOSTIC_PASS'
        receipt.update(scope='REAL_ENCODER_PUBLIC_LIBRARY_OUTPUT_DIAGNOSTIC',
                       real_encoder='EXECUTED_NOT_COMPONENT_ACCEPTED',
                       encoder_backend=args.real_encoder_backend,
                       full_asr='NOT_PROVEN_FEATURE_GATE_UNACCEPTED')
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.graph, server.backend, server.http_events = graph, args.backend, receipt["http_events"]
        server.extra_routes, server.extra_imports, server.entrypoint = routes, {"pako": "/vendor/pako.esm.mjs"}, "/recurrent.mjs"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="chrome", headless=True, args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"])
                try:
                    receipt["browser_version"] = browser.version
                    page = browser.new_page()
                    base = f"http://127.0.0.1:{server.server_port}"
                    page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(base + "/") else route.abort())
                    page.on("requestfailed", lambda request: receipt["browser_errors"].append({"url": request.url, "failure": request.failure}))
                    page.on("pageerror", lambda error: receipt["browser_errors"].append({"pageerror": str(error)}))
                    page.on("response", lambda response: receipt["browser_errors"].append({"url": response.url, "status": response.status}) if response.status >= 400 else None)
                    page.set_default_timeout(args.timeout_ms)
                    page.goto(base + "/")
                    page.wait_for_load_state("networkidle")
                    page.wait_for_function("typeof replayLibraryRecurrence === 'function'")
                    page.evaluate("args => { globalThis.pendingReplay=true; replayLibraryRecurrence(...args).then(r=>{globalThis.replayResult=r;globalThis.pendingReplay=false;}).catch(e=>{globalThis.replayResult={error:String(e)};globalThis.pendingReplay=false;}); }", [fixture, args.backend])
                    page.wait_for_function("pendingReplay === false")
                    receipt["result"] = page.evaluate("replayResult")
                finally:
                    browser.close()
            good = evaluate_result(receipt["result"], expected, args.backend)
            if args.trace_head:
                receipt["live_head"] = compare_live_head(
                    receipt["result"], head,
                    encoder_boundary="real" if args.real_encoder_backend else "frozen")
                receipt["intermediate_states_logits"] = receipt["live_head"]["status"]
                good = good and receipt["live_head"]["allclose"]
            receipt["status"] = success_status if good and not receipt["browser_errors"] else ('REAL_ENCODER_OUTPUT_DIAGNOSTIC_FAIL' if args.real_encoder_backend else 'CONDITIONAL_RECURRENCE_FAIL')
        except Exception as error:
            receipt["status"], receipt["error"] = "BROWSER_RECURRENCE_FAILED", str(error)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
            receipt["served_artifacts"] = []
            for event in receipt["http_events"]:
                if event["path"] == "/" or event.get("status") != 200:
                    continue
                path = resolve_route(event["path"], graph, routes)
                digest = sha256(path)
                exact = (event.get("body_complete") and event["body_sha256"] == digest
                         and digest == pinned.get(str(path.resolve())) and event["bytes_written"] == path.stat().st_size)
                receipt["served_artifacts"].append({"route": event["path"], "path": str(path), "sha256": digest, "delivered_exact": bool(exact)})
                if not exact:
                    receipt["status"] = "BROWSER_DELIVERY_FAILED"
            required = {"/recurrent.mjs", "/observe-session.mjs", "/graph.onnx", "/frozen-encoder.onnx", "/vocab.txt"}
            if args.real_encoder_backend:
                required.remove('/frozen-encoder.onnx')
                required.update({'/encoder-model.onnx', '/encoder-model.onnx.data'})
            if receipt["status"] == success_status and not required.issubset({item["route"] for item in receipt["served_artifacts"] if item["delivered_exact"]}):
                receipt["status"] = "REQUIRED_ARTIFACT_NOT_DELIVERED"
            for path, digest in pinned.items():
                if sha256(Path(path)) != digest:
                    receipt["status"] = "ARTIFACT_CHANGED_DURING_RUN"
            for item in inputs.values():
                if sha256(Path(item["path"])) != item["sha256"]:
                    receipt["status"] = "INPUT_CHANGED_DURING_RUN"
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output)}), flush=True)
    return 0 if receipt["status"] == success_status else 1


if __name__ == "__main__":
    raise SystemExit(main())
