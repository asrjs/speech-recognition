#!/usr/bin/env python3
"""
ASR.js Browser WebGPU Playwright Bridge.

Provides bidirectional orchestration between Node.js CLI / daemon workflows
and headless/headed Chrome WebGPU instances.
"""
import sys
import json
import time
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

COLD_AND_WARM_JS = """
async () => {
    const spec = __SPEC_JSON__;
    const mode = spec.mode || 'both';

    // 1. Cold Run (forceReload: true)
    let cold;
    if (mode === 'offline') {
        cold = await window.__ASR_HARNESS__.runOffline({ ...spec, forceReload: true });
    } else if (mode === 'streaming') {
        cold = await window.__ASR_HARNESS__.runStreaming({ ...spec, forceReload: true });
    } else {
        const off = await window.__ASR_HARNESS__.runOffline({ ...spec, forceReload: true });
        const str = await window.__ASR_HARNESS__.runStreaming({ ...spec, forceReload: false });
        cold = {
            schema: 'asrjs.browser-webgpu.v1',
            status: 'PASS',
            mode: 'both',
            modelId: off.modelId,
            backendId: off.backendId,
            adapter: off.adapter,
            limits: off.limits,
            transcript: off.transcript,
            offline: off,
            streaming: str,
            metrics: {
                loadMs: off.metrics.loadMs,
                offlineInferenceMs: off.metrics.inferenceMs,
                offlineRtfx: off.metrics.rtfx,
                streamingInferenceMs: str.metrics.inferenceMs,
                streamingRtfx: str.metrics.rtfx,
                ttftMs: str.metrics.ttftMs,
                totalGpuDispatches: (off.metrics.gpuDispatches ?? 0) + (str.metrics.gpuDispatches ?? 0)
            },
            timestamp: new Date().toISOString()
        };
    }

    // 2. Warm Run (forceReload: false, reuses in-memory session, loadMs: 0)
    let warm;
    if (mode === 'offline') {
        warm = await window.__ASR_HARNESS__.runOffline({ ...spec, forceReload: false });
    } else if (mode === 'streaming') {
        warm = await window.__ASR_HARNESS__.runStreaming({ ...spec, forceReload: false });
    } else {
        const off = await window.__ASR_HARNESS__.runOffline({ ...spec, forceReload: false });
        const str = await window.__ASR_HARNESS__.runStreaming({ ...spec, forceReload: false });
        warm = {
            schema: 'asrjs.browser-webgpu.v1',
            status: 'PASS',
            mode: 'both',
            modelId: off.modelId,
            backendId: off.backendId,
            adapter: off.adapter,
            limits: off.limits,
            transcript: off.transcript,
            offline: off,
            streaming: str,
            metrics: {
                loadMs: off.metrics.loadMs,
                offlineInferenceMs: off.metrics.inferenceMs,
                offlineRtfx: off.metrics.rtfx,
                streamingInferenceMs: str.metrics.inferenceMs,
                streamingRtfx: str.metrics.rtfx,
                ttftMs: str.metrics.ttftMs,
                totalGpuDispatches: (off.metrics.gpuDispatches ?? 0) + (str.metrics.gpuDispatches ?? 0)
            },
            timestamp: new Date().toISOString()
        };
    }

    return {
        cold,
        warm
    };
}
"""

BOTH_JS = """
async () => {
    const spec = __SPEC_JSON__;
    const offline = await window.__ASR_HARNESS__.runOffline(spec);
    const streaming = await window.__ASR_HARNESS__.runStreaming({ ...spec, forceReload: false });
    return {
        schema: 'asrjs.browser-webgpu.v1',
        status: 'PASS',
        mode: 'both',
        modelId: offline.modelId,
        backendId: offline.backendId,
        adapter: offline.adapter,
        limits: offline.limits,
        transcript: offline.transcript,
        offline: offline,
        streaming: streaming,
        metrics: {
            loadMs: offline.metrics.loadMs,
            offlineInferenceMs: offline.metrics.inferenceMs,
            offlineRtfx: offline.metrics.rtfx,
            streamingInferenceMs: streaming.metrics.inferenceMs,
            streamingRtfx: streaming.metrics.rtfx,
            ttftMs: streaming.metrics.ttftMs,
            totalGpuDispatches: (offline.metrics.gpuDispatches || 0) + (streaming.metrics.gpuDispatches || 0)
        },
        timestamp: new Date().toISOString()
    };
}
"""

def run_in_page(page, action, spec):
    # Wait for __ASR_HARNESS__ to be initialized on window
    page.wait_for_function('typeof window.__ASR_HARNESS__ !== "undefined"', timeout=60000)

    spec_json = json.dumps(spec)

    if action == 'probe':
        return page.evaluate('() => window.__ASR_HARNESS__.probeWebGpu()')

    elif action == 'catalog':
        return page.evaluate('() => window.__ASR_HARNESS__.getCatalog()')

    elif action == 'offline':
        code = f'() => window.__ASR_HARNESS__.runOffline({spec_json})'
        return page.evaluate(code)

    elif action == 'streaming':
        code = f'() => window.__ASR_HARNESS__.runStreaming({spec_json})'
        return page.evaluate(code)

    elif action == 'both':
        code = BOTH_JS.replace('__SPEC_JSON__', spec_json)
        return page.evaluate(code)

    elif action == 'cold_and_warm':
        code = COLD_AND_WARM_JS.replace('__SPEC_JSON__', spec_json)
        return page.evaluate(code)

    elif action == 'dispose':
        return page.evaluate('() => window.__ASR_HARNESS__.disposeAll()')

    else:
        raise ValueError(f'Unknown action: {action}')

def oneshot(harness_url, action, spec, timeout_ms=180000):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel='chrome',
            headless=True,
            args=CHROME_WEBGPU_ARGS
        )
        version = browser.version
        context = browser.new_context()
        page = context.new_page()
        page.on('console', log_console)
        page.on('pageerror', log_pageerror)

        print(f'[Bridge] Navigating to {harness_url}...', flush=True)
        page.goto(harness_url, timeout=60000)

        result = run_in_page(page, action, spec)
        if isinstance(result, dict):
            result['browser_version'] = version

        browser.close()
        return result

def connect_cdp(cdp_url, harness_url, action, spec, timeout_ms=180000):
    with sync_playwright() as p:
        print(f'[Bridge] Connecting over CDP to {cdp_url}...', flush=True)
        browser = p.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0] if browser.contexts else browser.new_context()
        
        # Look for existing harness page
        target_page = None
        for page in context.pages:
            if '/harness.html' in page.url or '127.0.0.1' in page.url:
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
