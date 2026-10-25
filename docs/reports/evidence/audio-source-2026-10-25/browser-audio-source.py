"""Real Chrome decode/resampling with a deliberately mocked model boundary."""
import argparse
import base64
import functools
import http.server
import io
import json
import math
from pathlib import Path
import struct
import threading
import wave
from playwright.sync_api import sync_playwright

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True, type=Path)
args = parser.parse_args()
root = args.root.resolve()
assert (root / 'dist/index.js').is_file()
audio = io.BytesIO()
with wave.open(audio, 'wb') as wav:
    wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
    wav.writeframes(b''.join(struct.pack('<h', int(1600 * math.sin(2 * math.pi * 440 * n / 16000))) for n in range(4000)))
fixture = audio.getvalue()
class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass
server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), functools.partial(Handler, directory=str(root)))
threading.Thread(target=server.serve_forever, daemon=True).start()
origin = f'http://127.0.0.1:{server.server_port}'
stub = """
export async function loadBuiltInSpeechModel() {
  window.receipt.loads++;
  return {
    runtime: {}, session: {},
    model: { info: { family: 'fixture', modelId: 'fixture', inference: {
      sampleRate: 8000, maxInputDurationSec: 30, recommendedWindowDurationSec: 30,
      minWindowDurationSec: 0.1, maxWindowDurationSec: 30, autoWindowThresholdSec: 30
    } } },
    async transcribe(audio) {
      if (audio.sampleRate !== 8000 || audio.numberOfFrames !== 2000) throw Error('Wrong prepared audio');
      window.receipt.inputs.push({sampleRate: audio.sampleRate, frames: audio.numberOfFrames});
      return {text: 'mock model boundary', warnings: [], meta: {}};
    },
    async dispose() { window.receipt.disposals++; }
  };
}
"""
try:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page()
        page.route(f'{origin}/dist/runtime/builtins.js', lambda route: route.fulfill(content_type='text/javascript', body=stub))
        page.route(f'{origin}/fixture.wav', lambda route: route.fulfill(content_type='audio/wav', body=fixture))
        page.route('https://huggingface.co/**', lambda route: route.abort())
        page.goto(origin, wait_until='networkidle')
        result = page.evaluate("""async ({origin, encoded}) => {
          window.receipt = {loads: 0, disposals: 0, inputs: [], contextsClosed: 0};
          const {transcribeSpeechFromAudioSource} = await import(origin + '/dist/index.js');
          const {createSpeechController} = await import(origin + '/dist/controller.js');
          const file = new File([Uint8Array.from(atob(encoded), c => c.charCodeAt(0))], 'tone.wav', {type:'audio/wav'});
          const audioOptions = {createAudioContext: rate => {
            const context = new AudioContext({sampleRate:rate});
            return {
              decodeAudioData: bytes => context.decodeAudioData(bytes),
              close: async () => {await context.close(); window.receipt.contextsClosed++;}
            };
          }};
          await transcribeSpeechFromAudioSource(file, {modelId:'fixture', audioOptions});
          await transcribeSpeechFromAudioSource(origin + '/fixture.wav', {modelId:'fixture', audioOptions});
          const controller = createSpeechController();
          await controller.load({modelId:'fixture'});
          await controller.transcribeSource(file, {}, audioOptions);
          await controller.transcribeSource(file, {}, audioOptions);
          if (controller.getSnapshot().status !== 'ready') throw Error('Controller did not become ready');
          await controller.dispose();
          let decodeFailed = false;
          try {
            await transcribeSpeechFromAudioSource(new Blob(['invalid audio']), {modelId:'fixture', audioOptions});
          } catch { decodeFailed = true; }
          if (!decodeFailed) throw Error('Invalid audio accepted');
          return {...window.receipt, decodeFailed, scope:'Real browser audio decoding; mocked model loading and inference. No weights.'};
        }""", {'origin': origin, 'encoded': base64.b64encode(fixture).decode()})
        assert result['loads'] == result['disposals'] == 4, result
        assert result['contextsClosed'] == 5, result
        assert len(result['inputs']) == 4, result
        browser.close()
    output = Path(__file__).parent / 'browser-audio-source.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(result, indent=2))
finally:
    server.shutdown()
    server.server_close()
