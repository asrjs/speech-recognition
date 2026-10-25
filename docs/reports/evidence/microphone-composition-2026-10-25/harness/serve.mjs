import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createSuiteServer } from '../../../demos/scripts/serve-suite.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const servers = {
  '/model/': createSuiteServer({ root: 'N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx' }),
  '/fixtures/': createSuiteServer({ root: path.resolve(here, '../../../speech-recognition/tests/fixtures') }),
};
const app = createSuiteServer({ root: path.join(here, 'dist') });
const server = http.createServer((request, response) => {
  const prefix = Object.keys(servers).find(prefix => request.url?.startsWith(prefix));
  if (prefix) {
    request.url = '/' + request.url.slice(prefix.length);
    servers[prefix].emit('request', request, response);
  } else app.emit('request', request, response);
});
server.listen(4342, '127.0.0.1', () => console.log('Mic acceptance on http://127.0.0.1:4342/'));
