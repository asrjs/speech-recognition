import { writeFile } from 'node:fs/promises';
import { listApprovedSpeechModelManifests } from '../../dist/index.js';

const [modelId, output, ...extra] = process.argv.slice(2);
if (!modelId || extra.length) {
  throw new Error('Usage: node tools/scripts/export-approved-model-manifest.mjs MODEL_ID [OUTPUT.json]');
}
const manifest = listApprovedSpeechModelManifests().find((item) => item.modelId === modelId);
if (!manifest) throw new Error(`No approved artifact manifest for "${modelId}".`);
const json = `${JSON.stringify(manifest, null, 2)}\n`;
if (output) await writeFile(output, json, { flag: 'wx' });
else process.stdout.write(json);
