"""Offline real-Chromium capture/controller and packaged TEN-VAD lifecycle smokes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
ORIGIN = "http://127.0.0.1:31847"
PROBE = Path(__file__).with_name("capture-lifecycle-probe.mjs")
PAKO = ROOT / "node_modules/pako/dist/pako.esm.mjs"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", type=Path, required=True, help="Installed Chromium/Chrome/Edge executable")
    parser.add_argument("--audio", type=Path, required=True, help="Local fake-device WAV fixture")
    parser.add_argument("--output", type=Path, required=True, help="New receipt path; existing files are never overwritten")
    parser.add_argument("--scenario", choices=["capture-owner", "microphone-controller", "ten-vad", "ten-vad-controller", "ten-vad-capture"], default="capture-owner")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("output must be a new file with an existing parent")
    ten_scenario = args.scenario in ("ten-vad", "ten-vad-controller", "ten-vad-capture")
    probe_names = {"ten-vad": "ten-vad-lifecycle-probe.mjs", "ten-vad-controller": "ten-vad-controller-probe.mjs", "ten-vad-capture": "capture-lifecycle-probe.mjs"}
    probe = Path(__file__).with_name(probe_names[args.scenario]) if ten_scenario else PROBE
    assets = [ROOT / f"dist/runtime/assets/ten-vad/ten_vad.{suffix}" for suffix in ("js", "wasm")] if ten_scenario else []
    for path in (args.browser, args.audio, ROOT / "dist/browser.js", probe, PAKO, *assets):
        if not path.is_file():
            parser.error(f"required file missing: {path}")
    # Import after preflight so --help and invalid arguments do not launch/install anything.
    from playwright.sync_api import sync_playwright
    dist = (ROOT / "dist").resolve()
    served = {}
    blocked = []
    page_errors = []
    source_paths = [probe, Path(__file__).resolve(), ROOT / "src/runtime/capture.ts", args.audio.resolve(), *assets]
    if ten_scenario:
        source_paths.extend(ROOT / f"src/runtime/{name}.ts" for name in ["ten-vad-browser", "ten-vad-worker", "ten-vad-protocol"])
    if args.scenario in ("microphone-controller", "ten-vad-controller", "ten-vad-capture"):
        source_paths.extend(ROOT / f"src/runtime/{name}.ts" for name in ["browser-controller", "browser-realtime", "browser-monitor"])
    if args.scenario in ("ten-vad-controller", "ten-vad-capture"):
        source_paths.extend(ROOT / f"src/runtime/{name}.ts" for name in ["streaming-detector", "streaming-config", "vad"])
    initial = {str(path): digest(path) for path in source_paths}

    def serve(route):
        url = urlsplit(route.request.url)
        if f"{url.scheme}://{url.netloc}" != ORIGIN:
            blocked.append(route.request.url)
            route.abort()
            return
        if url.path == "/":
            route.fulfill(content_type="text/html", body='<!doctype html><title>ASR capture lifecycle</title><script type="importmap">{"imports":{"pako":"/vendor/pako.esm.mjs"}}</script><body>Offline capture probe</body>')
            return
        explicit = {"/probe.mjs": probe, "/vendor/pako.esm.mjs": PAKO}
        if ten_scenario:
            explicit["/fixture.wav"] = args.audio.resolve()
            explicit.update({"/" + str(path.relative_to(ROOT)).replace("\\", "/"): path for path in assets})
        path = explicit.get(url.path, (ROOT / unquote(url.path).lstrip("/")).resolve())
        if path not in explicit.values() and (not path.is_relative_to(dist) or path.suffix != ".js"):
            blocked.append(route.request.url)
            route.abort()
            return
        if not path.is_file():
            blocked.append(route.request.url)
            route.abort()
            return
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if str(path) in served and served[str(path)] != sha:
            raise RuntimeError(f"served module changed: {path}")
        served[str(path)] = sha
        content_type = "application/wasm" if path.suffix == ".wasm" else "audio/wav" if path.suffix == ".wav" else "text/javascript"
        route.fulfill(content_type=content_type, body=data)

    result = {"status": "FAIL_BROWSER_LIFECYCLE", "scope": args.scenario, "runs": []}
    failure = None
    browser_closed = False
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(args.browser.resolve()), headless=True, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={args.audio.resolve()}",
            "--autoplay-policy=no-user-gesture-required",
        ])
        try:
            context = browser.new_context(permissions=["microphone"], service_workers="block")
            context.route("**/*", serve)
            page = context.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(ORIGIN, wait_until="networkidle", timeout=15000)
            # Start without awaiting evaluation so Playwright can enforce a wall-clock bound.
            probe_function = {"capture-owner": "runCaptureLifecycle", "microphone-controller": "runControllerLifecycle", "ten-vad": "runTenVadLifecycle", "ten-vad-controller": "runTenVadController", "ten-vad-capture": "runTenVadCapture"}[args.scenario]
            page.evaluate("fn => { window.probeResult = null; window.probeError = null; void import('/probe.mjs').then(m => m[fn]()).then(r => window.probeResult = r).catch(e => window.probeError = String(e.stack || e)); }", probe_function)
            page.wait_for_function("window.probeResult !== null || window.probeError !== null", timeout=30000)
            error = page.evaluate("window.probeError")
            if error:
                if ten_scenario:
                    result["progress"] = page.evaluate("window.tenVadProgress")
                raise RuntimeError(error)
            result = page.evaluate("window.probeResult")
            version = browser.version
            context.close()
        except Exception as error:
            failure = str(error)
        finally:
            browser.close()
            browser_closed = True
    if blocked or page_errors:
        failure = failure or f"blocked requests={blocked}; page errors={page_errors}"
    for path, sha in {**initial, **served}.items():
        if digest(Path(path)) != sha:
            failure = failure or f"input changed during smoke: {path}"
    if failure:
        result["status"] = "FAIL_BROWSER_LIFECYCLE"
    version = browser.version
    receipt = {**result, "browserVersion": version, "browserPath": str(args.browser.resolve()),
               "network": "PAGE_HTTP_REQUESTS_LOCALLY_FULFILLED_OR_REJECTED", "blockedRequests": blocked,
               "pageErrors": page_errors, "inputs": initial, "servedModules": served,
               "scenario": args.scenario, "error": failure, "browserClosed": browser_closed,
               "notValidated": ["physical microphone", "ASR", "WebGPU parity", "cleanup failure injection", "permission-dialog race"] + (["speech-detect controller"] if args.scenario not in ("ten-vad-controller", "ten-vad-capture") else ["model-driven segment boundaries"] + (["microphone capture"] if args.scenario == "ten-vad-controller" else [])) + (["VAD accuracy", "original-engine parity"] if ten_scenario else ["VAD"]) + (["browser controller"] if args.scenario == "capture-owner" else [])}
    with output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(receipt, target, indent=2)
        target.write("\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"], "runs": result["runs"], "receipt": str(output)}))
    return 1 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
