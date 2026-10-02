"""Offline real-Chromium capture/controller smokes; no sibling server or model loads."""
import argparse
import hashlib
import json
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
    parser.add_argument("--scenario", choices=["capture-owner", "microphone-controller"], default="capture-owner")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or not output.parent.is_dir():
        parser.error("output must be a new file with an existing parent")
    for path in (args.browser, args.audio, ROOT / "dist/browser.js", PROBE, PAKO):
        if not path.is_file():
            parser.error(f"required file missing: {path}")
    # Import after preflight so --help and invalid arguments do not launch/install anything.
    from playwright.sync_api import sync_playwright
    dist = (ROOT / "dist").resolve()
    served = {}
    blocked = []
    page_errors = []
    source_paths = [PROBE, Path(__file__).resolve(), ROOT / "src/runtime/capture.ts", args.audio.resolve()]
    if args.scenario == "microphone-controller":
        source_paths.extend(ROOT / f"src/runtime/{name}.ts" for name in ["browser-controller", "browser-realtime", "browser-monitor"])
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
        explicit = {"/probe.mjs": PROBE, "/vendor/pako.esm.mjs": PAKO}
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
        route.fulfill(content_type="text/javascript", body=data)

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
            probe_function = "runCaptureLifecycle" if args.scenario == "capture-owner" else "runControllerLifecycle"
            page.evaluate("fn => { window.probeResult = null; window.probeError = null; void import('/probe.mjs').then(m => m[fn]()).then(r => window.probeResult = r).catch(e => window.probeError = String(e.stack || e)); }", probe_function)
            page.wait_for_function("window.probeResult !== null || window.probeError !== null", timeout=30000)
            error = page.evaluate("window.probeError")
            if error:
                raise RuntimeError(error)
            result = page.evaluate("window.probeResult")
            version = browser.version
            context.close()
        finally:
            browser.close()
    if blocked or page_errors:
        raise RuntimeError(f"blocked requests={blocked}; page errors={page_errors}")
    for path, sha in {**initial, **served}.items():
        if digest(Path(path)) != sha:
            raise RuntimeError(f"input changed during smoke: {path}")
    receipt = {**result, "browserVersion": version, "browserPath": str(args.browser.resolve()),
               "network": "PAGE_HTTP_REQUESTS_LOCALLY_FULFILLED_OR_REJECTED", "blockedRequests": blocked,
               "pageErrors": page_errors, "inputs": initial, "servedModules": served,
               "scenario": args.scenario,
               "notValidated": ["physical microphone", "speech-detect controller", "VAD", "ASR", "WebGPU parity", "cleanup failure injection", "permission-dialog race"] + (["browser controller"] if args.scenario == "capture-owner" else [])}
    with output.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(receipt, target, indent=2)
        target.write("\n")
    print(json.dumps({"status": result["status"], "scope": result["scope"], "runs": result["runs"], "receipt": str(output)}))


if __name__ == "__main__":
    main()
