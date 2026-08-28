#!/usr/bin/env python3
"""Exercise the streaming-demo microphone and latency HUD with fake audio.

This is deliberately a browser-level acceptance probe rather than a unit test.
Chromium supplies a deterministic audio input device, while the demo still
uses its normal getUserMedia, capture, controller, and HUD paths.  Pass
``--model-dir`` to include a real local model transcription; without it the
probe validates capture and controller latency independently of model loading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


LATENCY_PATTERNS = {
    "first_partial_ms": re.compile(r"first partial\s+(\d+) ms"),
    "eou_ms": re.compile(r"eou\s+(\d+) ms"),
    "p50_process_ms": re.compile(r"p50 process\s+(\d+) ms"),
    "p95_emit_ms": re.compile(r"p95 emit\s+(\d+) ms"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("STREAMING_DEMO_URL", "http://127.0.0.1:3000/"))
    parser.add_argument("--audio", type=Path, required=True, help="WAV/PCM file used by Chromium's fake microphone")
    parser.add_argument("--model-dir", type=Path, help="Optional local Parakeet artifact directory")
    parser.add_argument("--model-id", default="parakeet-tdt-0.6b-v2")
    parser.add_argument("--backend", choices=("wasm", "webgpu-hybrid"), default="wasm")
    parser.add_argument("--mode", choices=("manual", "speech-detect"), default="manual")
    parser.add_argument("--capture-ms", type=int, default=3500)
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def select_with_option(page, value: str):
    selects = page.locator("select")
    for index in range(selects.count()):
        candidate = selects.nth(index)
        if candidate.locator(f"option[value='{value}']").count():
            candidate.select_option(value)
            return candidate
    raise AssertionError(f"No select containing option value {value!r}")


def select_runtime_backend(page, value: str):
    selects = page.locator("select")
    for index in range(selects.count()):
        candidate = selects.nth(index)
        if candidate.locator("option[value='webgpu-hybrid']").count():
            candidate.select_option(value)
            return candidate
    raise AssertionError("Runtime backend select was not rendered")


def set_directory_input(input_locator, directory: Path) -> int:
    files = [path for path in directory.rglob("*") if path.is_file()]
    if not files:
        raise AssertionError(f"Model directory is empty: {directory}")
    try:
        input_locator.set_input_files(str(directory))
        return len(files)
    except PlaywrightError:
        input_locator.set_input_files([str(path) for path in files])
        return len(files)


def write_result(result: dict[str, object], output: Path | None) -> None:
    serialized = json.dumps(result, indent=2, ensure_ascii=False)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{serialized}\n", encoding="utf-8")
    print(serialized)


def find_chromium_executable() -> str | None:
    configured = os.environ.get("CHROME_PATH")
    if configured:
        return configured
    candidates = [
        shutil.which("chrome"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next((candidate for candidate in candidates if candidate and Path(candidate).exists()), None)


def main() -> int:
    args = parse_args()
    audio = args.audio.resolve()
    if not audio.is_file():
        raise SystemExit(f"Audio fixture does not exist: {audio}")
    model_dir = args.model_dir.resolve() if args.model_dir else None
    if model_dir and not model_dir.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_dir}")

    console_messages: list[str] = []
    result: dict[str, object] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "url": args.url,
        "audio": str(audio),
        "model_dir": str(model_dir) if model_dir else None,
        "model_id": args.model_id if model_dir else None,
        "capture_ms": args.capture_ms,
        "browser": "chromium",
        "mode": args.mode,
        "backend": args.backend if model_dir else None,
        "audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
    }

    launch_kwargs: dict[str, object] = {
        "headless": True,
        "args": [
            "--use-fake-ui-for-media-stream",
            "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-audio-capture={audio}",
            "--autoplay-policy=no-user-gesture-required",
            "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding",
            "--no-sandbox",
        ],
    }
    executable = find_chromium_executable()
    if executable:
        launch_kwargs["executable_path"] = executable

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**launch_kwargs)
        context = browser.new_context(
            permissions=["microphone"],
            viewport={"width": 1440, "height": 1100},
        )
        page = context.new_page()
        page.on("console", lambda message: console_messages.append(f"{message.type}: {message.text}"))
        page.on("pageerror", lambda error: console_messages.append(f"pageerror: {error}"))
        page.add_init_script("window.localStorage.clear();")

        try:
            page.goto(args.url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            try:
                page.wait_for_load_state("networkidle", timeout=30_000)
            except PlaywrightTimeoutError:
                console_messages.append("warning: networkidle timeout; continuing after DOM load")
            page.get_by_role("button", name="Advanced", exact=True).click()
            select_with_option(page, args.mode)

            if model_dir:
                select_with_option(page, args.model_id)
                select_with_option(page, "local")
                select_runtime_backend(page, args.backend)
                folder_input = page.locator("input[type='file'][webkitdirectory]")
                file_count = set_directory_input(folder_input, model_dir)
                result["model_file_count"] = file_count
                page.get_by_role("button", name="Load Model").click()
                page.get_by_role("button", name="Model Loaded").wait_for(timeout=args.timeout_ms)
                result["model_loaded"] = True
            else:
                result["model_loaded"] = False

            page.get_by_role("button", name="Start Mic").click()
            page.get_by_role("button", name="Stop Mic").wait_for(timeout=30_000)
            page.wait_for_timeout(args.capture_ms)
            page.get_by_role("button", name="Stop Mic").click()
            page.get_by_role("button", name="Start Mic").wait_for(timeout=30_000)

            if model_dir:
                try:
                    page.get_by_text("Model ready", exact=True).wait_for(timeout=args.timeout_ms)
                except PlaywrightTimeoutError:
                    console_messages.append("warning: model-ready status did not reappear before timeout")

            page.get_by_text(re.compile(r"first partial\s+\d+ ms")).wait_for(timeout=args.timeout_ms)
            body_text = page.locator("body").inner_text()
            for field, pattern in LATENCY_PATTERNS.items():
                match = pattern.search(body_text)
                if not match:
                    raise AssertionError(f"HUD field {field} remained unavailable")
                result[field] = int(match.group(1))
            if "16000 Hz" not in body_text:
                raise AssertionError("HUD did not expose the 16 kHz processing capture rate")

            result["status"] = "passed"
            result["transcript_text"] = page.locator(".transcript-body").inner_text().strip()
            result["body_excerpt"] = " ".join(body_text.split())[-1200:]
            result["console_tail"] = console_messages[-40:]
            load_match = re.search(r"timeEnd: LoadModel: ([0-9.]+)", "\n".join(console_messages))
            if load_match:
                result["model_load_ms"] = float(load_match.group(1))
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=True)
                result["screenshot"] = str(args.screenshot.resolve())
        except Exception as error:
            result["status"] = "failed"
            result["error"] = str(error)
            result["console_tail"] = console_messages[-80:]
            try:
                result["body_excerpt"] = " ".join(page.locator("body").inner_text().split())[-1600:]
            except Exception:
                pass
            if args.screenshot:
                args.screenshot.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(args.screenshot), full_page=True)
                result["screenshot"] = str(args.screenshot.resolve())
            write_result(result, args.output)
            raise
        finally:
            context.close()
            browser.close()

    write_result(result, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
