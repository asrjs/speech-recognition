"""Matrix-only intervention on identical production power; not ASR acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from probe_captured_filterbank import captured_filterbank, digest
from replay_packed_fft import ROOT, read_capture, read_stage
from artifact_outputs import require_new_outputs, write_new_text
from tensor_parity import compare_tensors


NODE_PROBE = r'''
import {readFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {JSMelProcessor as ProductionProcessor} from './dist/audio/js-mel.js';
import {normalizeCandidate} from './tools/model-debugging/reference/hf-parakeet-port/normalization-reduction.mjs';
let input=''; for await(const chunk of process.stdin) input+=chunk;
const data=JSON.parse(input), original=readFileSync('./dist/audio/js-mel.js','utf8');
function replaceOnce(source,before,after) {
  if(source.split(before).length!==2) throw new Error('Processor source boundary differs');
  return source.replace(before,after);
}
const boundary='this.powerBuf[quarterN] = quarterRe * quarterRe + quarterIm * quarterIm;';
const results={};
for(const mode of ['baseline','actual_buffer','triangle_rounding']) {
  let source=replaceOnce(original,boundary,boundary+
    '\nfor(let bin=0;bin<N_FREQ_BINS;bin++) capturedPower[frameIndex*N_FREQ_BINS+bin]=this.powerBuf[bin];');
  source='const capturedPower = new Float32Array('+data.frames+'*257);\n'+source+
    '\nexport {capturedPower};';
  if(mode==='actual_buffer') source=replaceOnce(source,
    'this.melFilterbank = getCachedMelFilterbank(this.nMels);',
    'this.melFilterbank = Float32Array.from('+JSON.stringify(data.filterbank)+');');
  if(mode==='triangle_rounding') source=replaceOnce(source,
    'Math.max(0, Math.min(downSlope, upSlope)) * enorm',
    'Math.fround(Math.max(0, Math.min(downSlope, upSlope))) * enorm');
  const module=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
  const processor=new module.JSMelProcessor({nMels:128, validLengthMode:'onnx'});
  const raw=processor.computeRawMel(Float32Array.from(data.pcm));
  if(raw.nFrames!==data.frames || raw.validLength!==data.valid) throw new Error('Frame identity differs');
  results[mode]={raw:Array.from(raw.rawMel),
    features:Array.from(processor.normalizeFeatures(raw.rawMel,raw.nFrames,raw.validLength)),
    cascade_features:Array.from(normalizeCandidate(raw.rawMel,raw.nFrames,raw.validLength,8).features),
    power_sha256:createHash('sha256').update(Buffer.from(module.capturedPower.buffer)).digest('hex')};
}
if(new Set(Object.values(results).map(value=>value.power_sha256)).size!==1)
  throw new Error('Matrix intervention changed power');
const control=new ProductionProcessor({nMels:128, validLengthMode:'onnx'});
const rawControl=control.computeRawMel(Float32Array.from(data.pcm));
const featuresControl=control.normalizeFeatures(rawControl.rawMel,rawControl.nFrames,rawControl.validLength);
for(const [key,values] of [['raw',rawControl.rawMel],['features',featuresControl]])
  if(values.length!==results.baseline[key].length ||
     values.some((value,index)=>value!==results.baseline[key][index]))
    throw new Error('Instrumentation changed production output');
console.log(JSON.stringify(results));
'''


def evaluate(results, reference):
    names = ('baseline', 'actual_buffer', 'triangle_rounding')
    if set(results) != set(names):
        raise ValueError('Exactly three matrix conditions required')
    hashes = [results[name].get('power_sha256') for name in names]
    if (any(not isinstance(value, str) or len(value) != 64
            or any(char not in '0123456789abcdef' for char in value) for value in hashes)
            or len(set(hashes)) != 1):
        raise ValueError('Power identity missing or differs')
    tensors = {name: {key: read_stage({'stages': {key: {
        'dtype': 'float32', 'shape': list(reference.shape), 'data': results[name][key]}}}, key, 'float32')
        for key in ('raw', 'features', 'cascade_features')} for name in names}
    return {'power_sha256': hashes[0], 'feature_vs_official': {
        name: compare_tensors(reference, tensors[name]['features'], atol=1e-5, rtol=1e-4)
        for name in names}, 'cascade_feature_vs_official': {
        name: compare_tensors(reference, tensors[name]['cascade_features'], atol=1e-5, rtol=1e-4)
        for name in names}, 'matrix_effect': {
        name: {key: compare_tensors(tensors['baseline'][key], tensors[name][key], atol=0, rtol=0)
               for key in ('raw', 'features')} for name in names[1:]},
        'candidate_vs_actual': {key: compare_tensors(tensors['actual_buffer'][key],
            tensors['triangle_rounding'][key], atol=0, rtol=0) for key in ('raw', 'features')}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('capture-sha256', 'weight-sha256', 'sample-id'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    sources = [args.capture, Path(__file__), ROOT / 'dist/audio/js-mel.js',
               Path(__file__).with_name('normalization-reduction.mjs'),
               Path(__file__).with_name('probe_captured_filterbank.py'),
               Path(__file__).with_name('replay_packed_fft.py'),
               ROOT / 'tools/tensor_parity.py', ROOT / 'tools/artifact_outputs.py']
    require_new_outputs([args.output], protected_inputs=sources,
                        protected_directories=[ROOT / 'models', ROOT / 'src', ROOT / 'dist'])
    hashes = {str(path.resolve()): digest(path) for path in sources}
    meta, sample, pcm, _, _, reference, valid = read_capture(args.capture, args.capture_sha256, args.sample_id)
    if (meta.get('weight_sha256') != args.weight_sha256 or len(args.weight_sha256) != 64
            or not isinstance(meta.get('preprocessor_config'), dict)):
        raise ValueError('Original weight/config identity missing or differs')
    payload = {'pcm': pcm.ravel().tolist(), 'frames': reference.shape[2], 'valid': valid,
               'filterbank': captured_filterbank(sample).ravel().tolist()}
    process = subprocess.run(['node', '--input-type=module', '-e', NODE_PROBE], cwd=ROOT,
                             input=json.dumps(payload, allow_nan=False), text=True,
                             capture_output=True, check=True, timeout=60)
    metrics = evaluate(json.loads(process.stdout), reference)
    if hashes != {str(path.resolve()): digest(path) for path in sources}:
        raise ValueError('Diagnostic sources changed')
    receipt = {'status': 'FILTERBANK_FEATURE_DIAGNOSTIC_MEASURED',
        'scope': 'Matrix-only effect on identical production power; no feature waiver or ASR acceptance',
        'sample_id': args.sample_id, 'identity': meta, 'sources_sha256': hashes,
        'power_scope': 'Captured JS production power after packed FFT, not original NeMo power',
        'cascade_scope': 'Diagnostic width-8 f32 cascade on each unchanged raw boundary; not production normalization',
        'production_instrumentation_control': True,
        'node_probe_sha256': hashlib.sha256(NODE_PROBE.encode()).hexdigest(),
        'inputs_unchanged': True, **metrics}
    write_new_text(args.output, json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps(metrics))


if __name__ == '__main__':
    main()
