import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// Own ONNX Identity, float32[1], opset 17, IR 8. Lifecycle only, not an ASR graph.
// Generated with onnx.helper.make_model/make_graph/make_node('Identity').
const identity = Buffer.from(
  'CAgSFmFzcmpzLXJlc291cmNlLWZpeHR1cmU6WwoZCgVpbnB1dBIGb3V0cHV0IghJZGVudGl0eRITYXNyanNfcmVzb3VyY2Vfb25seVoTCgVpbnB1dBIKCggIARIECgIIAWIUCgZvdXRwdXQSCgoICAESBAoCCAFCBAoAEBE=',
  'base64',
);
export const resourceGraphSha256 =
  'b75fcc0af0e7502837826693e7abc70e0df19266fb324f8ca2d88e7e423e8fdc';

export async function writeNativeResourceFixture(directory: string) {
  if (createHash('sha256').update(identity).digest('hex') !== resourceGraphSha256) {
    throw new Error('Resource fixture identity mismatch');
  }
  const graph = join(directory, 'identity.onnx');
  const vocab = join(directory, 'vocab.txt');
  await writeFile(graph, identity, { flag: 'wx' });
  await writeFile(vocab, '▁resource 0\n<unk> 1\n', { flag: 'wx' });
  return { graph, vocab };
}
