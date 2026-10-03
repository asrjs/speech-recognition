"""Real packed fp32 browser encoder replay on frozen official features only."""
import argparse
import base64
import json
from pathlib import Path
import sys
import threading

import numpy as np

from browser_head_replay import ROOT, Handler, ThreadingHTTPServer, resolve_route
from export_nemo_fp32 import audit_graph, ENCODER_INPUTS, ENCODER_OUTPUTS, sha256
from validate_stage_replay import stage
from tensor_parity import compare_tensors
from artifact_outputs import require_new_outputs, write_new_text

sys.path.insert(0, str(ROOT / 'tools/model-debugging/reference/hf-parakeet-port'))
from replay_packed_fft import read_capture


def evaluate(result, sample, backend):
    if result.get('error') or result.get('cleanupErrors') or result.get('backendRequested') != backend:
        return False
    if result.get('crossOriginIsolated') is not True or len(result.get('runs', [])) != 2:
        return False
    good = True
    for repeat, run in enumerate(result['runs']):
        if run['repeat'] != repeat or set(run['outputs']) != set(ENCODER_OUTPUTS):
            raise ValueError('Actual encoder call/output contract differs')
        run['comparisons'] = {}
        for name, reference, dtype in (('outputs', 'encoder', 'float32'),
                                       ('encoded_lengths', 'encoder_length', 'int64')):
            run['comparisons'][name] = compare_tensors(stage(sample['stages'], reference, dtype),
                stage(run['outputs'], name, dtype), atol=1e-5, rtol=1e-4)
        good = all(value['allclose'] for value in run['comparisons'].values()) and good
        if backend == 'webgpu':
            good = run.get('gpuDispatches', 0) > 0 and good
    return good


def evaluate_sensitivity(result, sample, backend):
    """Measured downstream effect never waives a failed preprocessing gate."""
    runs = result.get('runs', [])
    if len(runs) != 4 or [run.get('featureSource') for run in runs] != ['official'] * 2 + ['production-js'] * 2:
        raise ValueError('Requested official/production repeated calls missing or reordered')
    if result.get('pcmSha256') != sample['model_input']['sha256']:
        raise ValueError('Actual browser PCM identity differs')
    if type(result.get('productionLength')) is not int or result['productionLength'] != sample['stages']['features_length']['data'][0]:
        raise ValueError('Production feature length differs')
    result['feature_comparison'] = compare_tensors(stage(sample['stages'], 'features', 'float32'),
        stage({'production': result['productionFeatures']}, 'production', 'float32'), atol=1e-5, rtol=1e-4)
    result['official_encoder_pass'] = evaluate({**result, 'runs': runs[:2]}, sample, backend)
    result['production_encoder_pass'] = evaluate({**result, 'runs': runs[2:]}, sample, backend)
    cross = result.get('crossSessionRuns') or []
    if cross:
        # Cross-session variability: same graph, same options, same features,
        # but a separately created session. Within-session repeats are bitwise
        # exact and cannot answer this.
        if [row.get('featureSource') for row in cross] != ['official', 'production-js']:
            raise ValueError('Cross-session feature sources differ from the main runs')
        pairs = (('official', runs[0], cross[0]), ('production-js', runs[2], cross[1]))
        comparisons = {}
        for label, first, second in pairs:
            if set(first['outputs']) != set(second['outputs']):
                raise ValueError('Cross-session output sets differ')
            comparisons[label] = {
                name: compare_tensors(np.asarray(first['outputs'][name]['data'], dtype=np.float32),
                                      np.asarray(second['outputs'][name]['data'], dtype=np.float32),
                                      atol=1e-5, rtol=1e-4)
                for name in first['outputs']
            }
        result['cross_session_comparison'] = comparisons
        result['cross_session_bitwise_identical'] = all(
            np.asarray(first['outputs'][name]['data'], dtype=np.float32).tobytes()
            == np.asarray(second['outputs'][name]['data'], dtype=np.float32).tobytes()
            for _, first, second in pairs
            for name in first['outputs']
        )
    resident = result.get('coResidentRuns') or []
    if resident:
        # Session co-residency: same encoder session, same graph, same options,
        # but a decoder session now exists alongside it.
        if [row.get('featureSource') for row in resident] != ['official', 'production-js']:
            raise ValueError('Co-resident feature sources differ from the main runs')
        resident_pairs = (('official', runs[0], resident[0]), ('production-js', runs[2], resident[1]))
        resident_comparisons = {}
        for label, first, second in resident_pairs:
            if set(first['outputs']) != set(second['outputs']):
                raise ValueError('Co-resident output sets differ')
            resident_comparisons[label] = {
                name: compare_tensors(np.asarray(first['outputs'][name]['data'], dtype=np.float32),
                                      np.asarray(second['outputs'][name]['data'], dtype=np.float32),
                                      atol=1e-5, rtol=1e-4)
                for name in first['outputs']
            }
        result['co_resident_comparison'] = resident_comparisons
        result['co_resident_bitwise_identical'] = all(
            np.asarray(first['outputs'][name]['data'], dtype=np.float32).tobytes()
            == np.asarray(second['outputs'][name]['data'], dtype=np.float32).tobytes()
            for _, first, second in resident_pairs
            for name in first['outputs']
        )
    return result['official_encoder_pass']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'encoder', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'encoder-sha256', 'encoder-data-sha256', 'sample-id'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--backend', choices=['webgpu', 'wasm'], required=True)
    parser.add_argument('--production-features', action='store_true', help='Measure actual JS feature sensitivity alongside the unchanged official-feature control')
    parser.add_argument('--cross-session', action='store_true', help='Opt-in: create a second identical encoder session and compare, testing cross-session variability that within-session repeats cannot cover')
    parser.add_argument('--co-resident-decoder', action='store_true', help='Opt-in: hold a decoder session alongside the encoder and re-run it, testing whether session co-residency changes encoder output')
    parser.add_argument('--create-as-encoder-name', action='store_true', help='Opt-in: create the encoder session from /encoder-model.onnx instead of /graph.onnx, the same bytes under the library artifact name, to test whether the create-URL name affects the result')
    parser.add_argument('--timeout-ms', type=int, default=240000)
    args = parser.parse_args()
    if not 1000 <= args.timeout_ms <= 600000:
        parser.error('Bounded timeout must be 1000..600000ms')
    data = Path(str(args.encoder) + '.data')
    module = Path(__file__).with_suffix('.mjs')
    sources = [args.capture, args.encoder, data, Path(__file__), module,
        Path(__file__).with_name('browser_head_replay.py'), ROOT / 'dist/models/nemo-tdt/ort.js']
    if args.production_features:
        sources.append(ROOT / 'dist/audio/js-mel.js')
    decoder_path = None
    if args.co_resident_decoder:
        # Resolved before hashing so the served graph is bound by sources_sha256
        # and inputs_unchanged, not added after the fact.
        decoder_path = ROOT / 'models/orukeet/onnx/fp32-packed-2026-10-01/decoder_joint-model.onnx'
        if not decoder_path.is_file():
            raise ValueError('Requested co-resident decoder graph is missing')
        sources.append(decoder_path)
    require_new_outputs([args.output], protected_inputs=sources,
                        protected_directories=[ROOT / 'models', ROOT / 'src', ROOT / 'dist'])
    hashes = {str(path.resolve()): sha256(path) for path in sources}
    for path, expected in ((args.capture, args.capture_sha256), (args.encoder, args.encoder_sha256),
                           (data, args.encoder_data_sha256)):
        if hashes[str(path.resolve())] != expected:
            raise ValueError('Pinned source differs: ' + str(path))
    artifact = audit_graph(args.encoder, ENCODER_INPUTS, ENCODER_OUTPUTS)
    if len(artifact['external_data']) != 1 or artifact['external_data'][0]['location'] != 'encoder-model.onnx.data':
        raise ValueError('Only own packed encoder-model.onnx.data contract supported')
    meta, sample, pcm, _, _, features, valid = read_capture(args.capture, args.capture_sha256, args.sample_id)
    fixture = {'featuresF32le': base64.b64encode(features.astype('<f4').tobytes()).decode('ascii'),
               'shape': list(features.shape), 'length': valid}
    if args.cross_session:
        fixture['crossSession'] = True
    if args.create_as_encoder_name:
        fixture['createAsEncoderName'] = True
    if args.co_resident_decoder:
        fixture['coResidentDecoder'] = True
    if args.production_features:
        fixture.update(pcmF32le=base64.b64encode(pcm.astype('<f4').tobytes()).decode('ascii'),
                       pcmSha256=sample['model_input']['sha256'])
    receipt = {'status': 'BROWSER_ENCODER_COMPONENT_FAIL', 'scope': 'Official frozen features through real fp32 encoder/library session bridge; preprocessing, decoder, public ASR and every-node GPU placement untested',
        'sources_sha256': hashes, 'identity': meta, 'graph': artifact, 'sample_id': args.sample_id,
        'backend': args.backend, 'browser_errors': [], 'blocked_requests': []}
    receipt['production_features_requested'] = args.production_features
    if args.production_features:
        receipt['scope'] = 'Actual browser JS feature -> real fp32 encoder sensitivity with official-feature baseline; failed feature gate remains failed, decoder/full ASR untested'
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    server.graph, server.backend = args.encoder, args.backend
    server.entrypoint = '/encoder-replay.mjs'
    server.extra_routes = {'/encoder-replay.mjs': module, '/encoder-model.onnx.data': data,
                           '/observe-session.mjs': Path(__file__).with_name('observe_session.mjs')}
    # The same graph bytes are also served under the library's own artifact name,
    # so the effect of the create-URL name can be measured on identical bytes.
    server.extra_routes['/encoder-model.onnx'] = args.encoder
    if decoder_path is not None:
        server.extra_routes['/decoder-graph.onnx'] = decoder_path
        receipt['scope'] += '; co-resident decoder session requested'
    server.http_events = receipt['http_events'] = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{server.server_port}'
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel='chrome', headless=True,
                args=['--enable-unsafe-webgpu'])
            try:
                receipt['browser_version'] = browser.version
                context = browser.new_context(service_workers='block')
                def route(request_route):
                    request = request_route.request
                    try:
                        if request.method != 'GET' or not request.url.startswith(origin + '/'):
                            raise ValueError('Outside owned origin')
                        if request.url != origin + '/':
                            resolve_route(request.url, args.encoder, server.extra_routes)
                        request_route.continue_()
                    except ValueError:
                        receipt['blocked_requests'].append(request.url)
                        request_route.abort()
                context.route('**/*', route)
                page = context.new_page()
                page.on('pageerror', lambda error: receipt['browser_errors'].append(str(error)))
                page.on('requestfailed', lambda request: receipt['browser_errors'].append(request.url))
                page.set_default_timeout(args.timeout_ms)
                page.goto(origin + '/', wait_until='networkidle')
                page.wait_for_function("typeof replayEncoder === 'function'")
                page.evaluate("args => {globalThis.done=false; replayEncoder(...args).then(r=>{globalThis.result=r;globalThis.done=true;}).catch(e=>{globalThis.result={error:String(e)};globalThis.done=true;});}", [fixture, args.backend])
                page.wait_for_function('done === true')
                result = receipt['result'] = page.evaluate('result')
                measured = evaluate_sensitivity(result, sample, args.backend) if args.production_features else evaluate(result, sample, args.backend)
                if measured and not receipt['browser_errors'] and not receipt['blocked_requests']:
                    receipt['status'] = 'BROWSER_ENCODER_SENSITIVITY_MEASURED' if args.production_features else 'BROWSER_ENCODER_COMPONENT_PASS'
                context.close()
            finally:
                browser.close()
    except Exception as error:
        receipt['status'] = 'BROWSER_ENCODER_COMPONENT_FAIL'
        receipt['error'] = str(error)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        receipt['server_stopped'] = not thread.is_alive()
        receipt['inputs_unchanged'] = hashes == {str(path.resolve()): sha256(path) for path in sources}
        receipt['artifact_delivery_verified'] = all(any(
            event.get('path') == route and event.get('status') == 200 and event.get('body_complete') is True
            and event.get('bytes_written') == path.stat().st_size
            and event.get('body_sha256') == hashes[str(path.resolve())]
            for event in receipt['http_events'])
            for route, path in (
                (('/encoder-model.onnx' if args.create_as_encoder_name else '/graph.onnx'), args.encoder),
                ('/encoder-model.onnx.data', data)))
        if not receipt['inputs_unchanged'] or not receipt['server_stopped'] or not receipt['artifact_delivery_verified']:
            receipt['status'] = 'BROWSER_ENCODER_COMPONENT_FAIL'
        write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': receipt['status'], 'output': str(args.output), 'error': receipt.get('error'),
                      'browser_error': receipt.get('result', {}).get('error')}))
    return 0 if receipt['status'] in {'BROWSER_ENCODER_COMPONENT_PASS', 'BROWSER_ENCODER_SENSITIVITY_MEASURED'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
