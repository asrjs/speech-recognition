import http from 'node:http';
import { createSuiteServer } from '../../../demos/scripts/serve-suite.mjs';
const app = createSuiteServer({ root: 'N:/github/asrjs/demos/dist' });
const model = createSuiteServer({ root: 'N:/models/onnx/nemo/parakeet-tdt-0.6b-v3-onnx' });
const server = http.createServer((req,res) => {
  if (req.url?.startsWith('/model/')) { req.url = '/' + req.url.slice('/model/'.length); model.emit('request',req,res); }
  else app.emit('request',req,res);
});
server.listen(4343,'127.0.0.1',()=>console.log('Approved-profile acceptance: http://127.0.0.1:4343/'));
