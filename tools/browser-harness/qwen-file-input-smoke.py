#!/usr/bin/env python3
"""Run a Qwen local-entry smoke using browser File objects for selected assets."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=r"N:\github\asrjs\speech-recognition")
    parser.add_argument("--model-dir", default=r"N:\models\onnx\qwen3-asr-0.6b-official")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--port", type=int, default=8996)
    parser.add_argument("--case", action="append", default=None,
                        help="Fixture label=URL; repeat to run multiple fixtures in one browser session")
    parser.add_argument("--timeout-ms", type=int, default=900000)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cases = args.case or ["jfk-short.wav=/__fixtures__/jfk-short.wav"]
    fixtures = []
    for case in cases:
        label, separator, audio_url = case.partition("=")
        if not separator or not label or not audio_url:
            raise SystemExit(f"Invalid --case {case!r}; expected label=URL")
        fixtures.append({"fixture": label, "audioUrl": audio_url})

    repo = Path(args.repo)
    model_dir = Path(args.model_dir)
    server_script = repo / "tools" / "browser-harness" / "server.mjs"
    if not server_script.is_file():
        raise SystemExit(f"Server script not found: {server_script}")
    selected_entries = [
        ("audio-encoder-dynamic.onnx", model_dir / "audio-encoder-dynamic.onnx"),
        ("decoder-prefill-fp16.onnx", model_dir / "decoder-prefill-fp16.onnx"),
        ("decoder-step-fp16.onnx", model_dir / "decoder-step-fp16.onnx"),
        ("decoder-fp16.onnx.data", model_dir / "decoder-fp16.onnx.data"),
        ("tokenizer/tokenizer.json", model_dir / "tokenizer" / "tokenizer.json"),
    ]
    for _, file_path in selected_entries:
        if not file_path.is_file():
            raise SystemExit(f"Required selected model artifact not found: {file_path}")
    model_id = args.model_id or model_dir.name.removesuffix("-official")
    local_folder = f"__external_models__/{model_dir.name}"

    env = os.environ.copy()
    env["ASRJS_MODELS_DIR"] = str(model_dir.parent)
    server = subprocess.Popen(
        ["node", str(server_script), "--port", str(args.port)],
        cwd=str(repo), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        health_url = f"http://127.0.0.1:{args.port}/__api__/health"
        import urllib.request
        for _ in range(60):
            if server.poll() is not None:
                raise RuntimeError("Harness server exited: " + (server.stdout.read() or ""))
            try:
                with urllib.request.urlopen(health_url, timeout=1):
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("Harness server did not become ready")

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                channel="chrome", headless=True,
                args=["--enable-unsafe-webgpu", "--ignore-gpu-blocklist",
                      "--use-angle=d3d11", "--enable-features=Vulkan"],
            )
            try:
                page = browser.new_page()
                page.on("console", lambda msg: print(f"[browser console] {msg.text}", flush=True))
                page.on("requestfailed", lambda req: print(
                    f"[browser request failed] {req.url}: {req.failure}", file=sys.stderr, flush=True))
                page.on("pageerror", lambda err: print(f"[browser pageerror] {err}", file=sys.stderr, flush=True))
                page.goto(f"http://127.0.0.1:{args.port}/harness.html", timeout=60000)
                page.evaluate("""() => {
                    const input = document.createElement('input');
                    input.type = 'file'; input.multiple = true; input.id = 'qwen-file-input';
                    input.hidden = true; document.body.appendChild(input);
                }""")
                page.locator("#qwen-file-input").set_input_files(
                    [str(path) for _, path in selected_entries])
                page.evaluate("""(manifest) => {
                    const files = new Map([...document.querySelector('#qwen-file-input').files]
                      .map(file => [file.name, file]));
                    window.__qwenLocalEntries = manifest.map(({ path, name }) => {
                      const file = files.get(name);
                      if (!file) throw new Error(`Playwright File missing for ${path}`);
                      return {
                        path, basename: name.split('/').pop(), size: file.size,
                        lastModified: file.lastModified,
                        handle: { kind: 'file', async getFile() { return file; } },
                      };
                    });
                }""", [{"path": rel, "name": path.name} for rel, path in selected_entries])
                page.wait_for_function(
                    "typeof window.__ASR_HARNESS__ !== 'undefined'", timeout=60000)
                base_spec = {
                    "localFolder": local_folder,
                    "modelId": model_id,
                    "backend": "wasm",
                    "options": {"decoderQuant": "fp16", "cpuThreads": 1},
                    "mode": "offline",
                }
                result = page.evaluate("""async ({spec, timeoutMs}) => {
                    const results = [];
                    let fatalError = null;
                    let cleanup = { status: 'EXECUTED', error: null };
                    let runAction;
                    try {
                      ({ runAction } = await import('/tools/browser-harness/runner.mjs'));
                      for (let index = 0; index < spec.fixtures.length; index += 1) {
                        let timer;
                        const current = { ...spec.baseSpec, ...spec.fixtures[index],
                          localEntries: window.__qwenLocalEntries, forceReload: index === 0 };
                        const execution = runAction(window.__ASR_HARNESS__, 'offline', current);
                        try {
                          const run = await Promise.race([
                            execution,
                            new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Timed out')),
                              timeoutMs); }),
                          ]);
                          results.push({ fixture: current.fixture, audioUrl: current.audioUrl, status: 'EXECUTED', run });
                        } catch (error) {
                          results.push({ fixture: current.fixture, audioUrl: current.audioUrl, status: 'FAIL',
                            error: String(error?.message ?? error) });
                          break;
                        } finally { clearTimeout(timer); }
                      }
                    } catch (error) {
                      fatalError = String(error?.message ?? error);
                    } finally {
                      try {
                        if (!runAction) ({ runAction } = await import('/tools/browser-harness/runner.mjs'));
                        if (window.__ASR_HARNESS__) await Promise.race([
                          runAction(window.__ASR_HARNESS__, 'dispose', {}),
                          new Promise((_, reject) => setTimeout(() => reject(new Error('Cleanup timed out')), 10000)),
                        ]);
                      } catch (error) {
                        cleanup = { status: 'FAIL', error: String(error?.message ?? error) };
                      }
                    }
                    return { runs: results, cleanup, fatalError };
                }""", {"spec": {"baseSpec": base_spec, "fixtures": fixtures}, "timeoutMs": args.timeout_ms})
                output = {
                    "status": "EXECUTED" if result["cleanup"]["status"] == "EXECUTED"
                        and not result["fatalError"]
                        and all(item["status"] == "EXECUTED" for item in result["runs"]) else "FAIL",
                    "scope": "public local-entry load and offline decodes; numerical parity unverified",
                    "reuse": "single loaded handle across fixtures in one browser page",
                    "cleanup": result["cleanup"],
                    "fatalError": result["fatalError"],
                    "provenance": {
                        "sourceHead": subprocess.check_output(
                            ["git", "-c", f"safe.directory={repo.as_posix()}", "rev-parse", "HEAD"], cwd=str(repo), text=True).strip(),
                        "trackedDiffSha256": sha256_bytes(subprocess.check_output(
                            ["git", "-c", f"safe.directory={repo.as_posix()}", "diff", "--binary", "HEAD"], cwd=str(repo))),
                        "scriptSha256": sha256_file(Path(__file__).resolve()),
                        "onnxruntimeWeb": json.loads((repo / "node_modules" / "onnxruntime-web" /
                            "package.json").read_text(encoding="utf-8"))["version"],
                        "browserVersion": browser.version,
                        "modelArtifacts": [
                            {"path": rel, "sha256": sha256_file(path)}
                            for rel, path in selected_entries
                        ],
                    },
                    "filesSuppliedAsBrowserFile": [
                        {"path": rel, "size": path.stat().st_size}
                        for rel, path in selected_entries
                    ],
                    "runs": result["runs"],
                }
                write_result(output, args.out)
                if output["status"] != "EXECUTED":
                    return 1
            finally:
                browser.close()
    except FileExistsError as error:
        print(f"Cannot overwrite existing receipt: {error.filename}", file=sys.stderr)
        return 1
    except Exception as error:
        try:
            write_result({
                "status": "FAIL",
                "scope": "public local-entry load; numerical parity unverified",
                "cleanup": {"status": "FAIL", "error": "browser run did not report cleanup"},
                "fixtures": fixtures,
                "error": str(error),
            }, args.out)
        except FileExistsError as receipt_error:
            print(f"Could not preserve FAIL receipt without overwriting: {receipt_error.filename}",
                  file=sys.stderr)
        return 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return 0


def write_result(result, out_path):
    if out_path is None:
        output_dir = Path(__file__).resolve().parents[1] / "data" / "results" / "browser-harness"
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        out_path = output_dir / f"qwen-file-input-{timestamp}.json"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print("RESULT_JSON:" + json.dumps(result, ensure_ascii=False))
    print(f"RESULT_FILE:{out_path}")


def sha256_file(file_path):
    digest = hashlib.sha256()
    with Path(file_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
