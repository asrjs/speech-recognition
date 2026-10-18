#!/usr/bin/env python3
"""
ASR.js Browser WebGPU Playwright Bridge.

Provides bidirectional orchestration between Node.js CLI / daemon workflows
and headless/headed Chrome WebGPU instances.
"""
import sys
import json
from playwright.sync_api import sync_playwright

CHROME_WEBGPU_ARGS = [
    '--enable-unsafe-webgpu',
    '--ignore-gpu-blocklist',
    '--use-angle=d3d11',
    '--enable-features=Vulkan',
]

def log_console(msg):
    text = msg.text
    print(f'[browser console] {text}', flush=True)

def log_pageerror(err):
    print(f'[browser pageerror] {err}', file=sys.stderr, flush=True)

def run_in_page(page, action, spec):
    page.wait_for_function('typeof window.__ASR_HARNESS__ !== "undefined"', timeout=60000)
    return page.evaluate("""async ({action, spec}) => {
        const {runAction} = await import('/tools/browser-harness/runner.mjs');
        let timer;
        try {
            return await Promise.race([
                runAction(window.__ASR_HARNESS__, action, spec),
                new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Browser action timed out')), spec.timeoutMs || 180000); })
            ]);
        } finally { clearTimeout(timer); }
    }""", {"action": action, "spec": spec})


def oneshot(harness_url, action, spec, timeout_ms=180000):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel='chrome',
            headless=True,
            args=CHROME_WEBGPU_ARGS
        )
        try:
            version = browser.version
            context = browser.new_context()
            page = context.new_page()
            page.on('console', log_console)
            page.on('pageerror', log_pageerror)
            page.on('requestfailed', lambda request: print(f'[browser request failed] {request.url}: {request.failure}', file=sys.stderr, flush=True))

            print(f'[Bridge] Navigating to {harness_url}...', flush=True)
            page.goto(harness_url, timeout=60000)

            result = run_in_page(page, action, spec)
            if isinstance(result, dict):
                result['browser_version'] = version

            return result
        finally:
            browser.close()


def connect_cdp(cdp_url, harness_url, action, spec, timeout_ms=180000):
    with sync_playwright() as p:
        print(f'[Bridge] Connecting over CDP to {cdp_url}...', flush=True)
        browser = p.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()

        # Look for existing harness page
        target_page = None
        for page in context.pages:
            if page.url == harness_url:
                target_page = page
                break

        if not target_page:
            target_page = context.new_page()
            target_page.goto(harness_url, timeout=60000)

        target_page.on('console', log_console)
        target_page.on('pageerror', log_pageerror)

        result = run_in_page(target_page, action, spec)
        if isinstance(result, dict):
            result['browser_version'] = browser.version
            result['cdp_warm_connected'] = True

        return result

def main():
    if len(sys.argv) < 5:
        print('Usage: bridge.py <oneshot|cdp> <url|cdp_url> <action> <spec_json> [harness_url]')
        sys.exit(1)

    mode = sys.argv[1]
    target_url = sys.argv[2]
    action = sys.argv[3]
    spec = json.loads(sys.argv[4])
    harness_url = sys.argv[5] if len(sys.argv) > 5 else target_url

    try:
        if mode == 'oneshot':
            result = oneshot(target_url, action, spec)
        elif mode == 'cdp':
            result = connect_cdp(target_url, harness_url, action, spec)
        else:
            raise ValueError(f'Invalid mode: {mode}')

        print('RESULT_PAYLOAD:' + json.dumps(result))
    except Exception as e:
        print(f'[Bridge Error] {e}', file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
