#!/usr/bin/env node
/**
 * ASR.js Browser WebGPU Test Harness Server.
 *
 * Provides high-performance static file serving with COOP/COEP isolation headers,
 * Range request support for large ONNX weights, and discovery APIs for local
 * models and audio fixtures.
 */
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, '..', '..');

const MIME_MAP = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.wasm': 'application/wasm',
  '.json': 'application/json; charset=utf-8',
  '.onnx': 'application/octet-stream',
  '.data': 'application/octet-stream',
  '.bin': 'application/octet-stream',
  '.npy': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
  '.wav': 'audio/wav',
  '.ogg': 'audio/ogg',
  '.mp3': 'audio/mpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

export function resolveSafePath(rootDir, relativePath) {
  const normalized = path.normalize(decodeURIComponent(relativePath)).replace(/^(\.\.[/\\])+/, '');
  const resolved = path.resolve(rootDir, normalized);
  if (!resolved.startsWith(path.resolve(rootDir))) {
    return null;
  }
  return resolved;
}

export function createHarnessServer(options = {}) {
  const port = options.port || 8999;
  const externalModelsDir = process.env.ASRJS_MODELS_DIR || 'N:/models';
  const fixturesDir = path.join(REPO_ROOT, 'tools', 'data', 'fixtures', 'audio');
  const modelsDir = path.join(REPO_ROOT, 'models', 'onnx');

  const server = http.createServer((req, res) => {
    // Enable CORS and COOP/COEP for SharedArrayBuffer and WebGPU
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, HEAD');
    res.setHeader('Access-Control-Allow-Headers', '*');
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');

    if (req.method === 'OPTIONS') {
      res.writeHead(204);
      res.end();
      return;
    }

    const parsedUrl = new URL(req.url, `http://${req.headers.host || '127.0.0.1'}`);
    const pathname = parsedUrl.pathname;

    // API: /__api__/health
    if (pathname === '/__api__/health' || pathname === '/__api__/ping') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', timestamp: Date.now(), port }));
      return;
    }

    // API: /__api__/local-models
    if (pathname === '/__api__/local-models') {
      try {
        const results = [];
        if (fs.existsSync(modelsDir)) {
          const entries = fs.readdirSync(modelsDir, { withFileTypes: true });
          for (const entry of entries) {
            if (entry.isDirectory()) {
              const fullDir = path.join(modelsDir, entry.name);
              const files = fs.readdirSync(fullDir);
              let totalBytes = 0;
              for (const f of files) {
                try {
                  const st = fs.statSync(path.join(fullDir, f));
                  if (st.isFile()) totalBytes += st.size;
                } catch {}
              }
              results.push({
                id: entry.name,
                name: entry.name,
                folder: `models/onnx/${entry.name}`,
                dir: `models/onnx/${entry.name}`,
                fileCount: files.length,
                files,
                sizeBytes: totalBytes,
                external: false,
              });
            }
          }
        }
        if (fs.existsSync(externalModelsDir) && path.resolve(externalModelsDir) !== path.resolve(modelsDir)) {
          try {
            const extEntries = fs.readdirSync(externalModelsDir, { withFileTypes: true });
            for (const entry of extEntries) {
              if (entry.isDirectory()) {
                const fullDir = path.join(externalModelsDir, entry.name);
                const files = fs.readdirSync(fullDir);
                let totalBytes = 0;
                for (const f of files) {
                  try {
                    const st = fs.statSync(path.join(fullDir, f));
                    if (st.isFile()) totalBytes += st.size;
                  } catch {}
                }
                results.push({
                  id: `external-${entry.name}`,
                  name: `[External] ${entry.name}`,
                  folder: `__external_models__/${entry.name}`,
                  dir: `__external_models__/${entry.name}`,
                  fileCount: files.length,
                  files,
                  sizeBytes: totalBytes,
                  external: true,
                });
              }
            }
          } catch {}
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(results, null, 2));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // API: /__api__/local-entries?dir=<dir>
    if (pathname === '/__api__/local-entries') {
      const requestedDir = parsedUrl.searchParams.get('dir');
      if (!requestedDir) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Missing "dir" query parameter' }));
        return;
      }
      try {
        let targetDir;
        let urlPrefix;
        if (requestedDir.startsWith('__external_models__/')) {
          const sub = requestedDir.replace(/^__external_models__\//, '');
          targetDir = resolveSafePath(externalModelsDir, sub);
          urlPrefix = `/__external_models__/${sub}`;
        } else if (requestedDir.startsWith('models/onnx/')) {
          const sub = requestedDir.replace(/^models\/onnx\//, '');
          targetDir = resolveSafePath(modelsDir, sub);
          urlPrefix = `/models/onnx/${sub}`;
        } else if (requestedDir.startsWith('/models/onnx/')) {
          const sub = requestedDir.replace(/^\/models\/onnx\//, '');
          targetDir = resolveSafePath(modelsDir, sub);
          urlPrefix = `/models/onnx/${sub}`;
        } else {
          targetDir = resolveSafePath(modelsDir, requestedDir);
          urlPrefix = `/models/onnx/${requestedDir}`;
        }

        if (!targetDir || !fs.existsSync(targetDir) || !fs.statSync(targetDir).isDirectory()) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: `Directory not found: ${requestedDir}` }));
          return;
        }

        const entries = [];
        const files = fs.readdirSync(targetDir);
        for (const file of files) {
          const fullPath = path.join(targetDir, file);
          const st = fs.statSync(fullPath);
          if (st.isFile()) {
            entries.push({
              name: file,
              path: file,
              relativePath: file,
              size: st.size,
              url: `${urlPrefix}/${file}`.replace(/\/+/g, '/'),
            });
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(entries, null, 2));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // API: /__api__/fixtures
    if (pathname === '/__api__/fixtures') {
      try {
        const fixtures = [];
        if (fs.existsSync(fixturesDir)) {
          const files = fs.readdirSync(fixturesDir);
          for (const file of files) {
            const st = fs.statSync(path.join(fixturesDir, file));
            if (st.isFile() && (file.endsWith('.wav') || file.endsWith('.ogg'))) {
              fixtures.push({
                name: file,
                path: file,
                size: st.size,
                url: `/__fixtures__/${file}`,
              });
            }
          }
        }
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(fixtures, null, 2));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
      return;
    }

    // Route: harness.html or root /
    if (pathname === '/' || pathname === '/harness.html') {
      const htmlPath = path.join(__dirname, 'harness.html');
      if (!fs.existsSync(htmlPath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('harness.html not found');
        return;
      }
      serveFile(req, res, htmlPath, '.html');
      return;
    }

    // Route: /__fixtures__/*
    if (pathname.startsWith('/__fixtures__/')) {
      const rel = pathname.substring('/__fixtures__/'.length);
      const filePath = resolveSafePath(fixturesDir, rel);
      if (!filePath || !fs.existsSync(filePath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end(`Fixture not found: ${pathname}`);
        return;
      }
      serveFile(req, res, filePath);
      return;
    }

    // Route: /__external_models__/*
    if (pathname.startsWith('/__external_models__/')) {
      const rel = pathname.substring('/__external_models__/'.length);
      const filePath = resolveSafePath(externalModelsDir, rel);
      if (!filePath || !fs.existsSync(filePath)) {
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end(`External model not found: ${pathname}`);
        return;
      }
      serveFile(req, res, filePath);
      return;
    }

    // Default static resolution relative to REPO_ROOT:
    // /dist/*, /node_modules/*, /models/onnx/*, /.cache/*, /tools/*
    const filePath = resolveSafePath(REPO_ROOT, pathname.replace(/^\/+/, ''));
    if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
      console.log('[Server 404]', req.method, pathname);
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      res.end(`File Not Found: ${pathname}`);
      return;
    }

    serveFile(req, res, filePath);
  });

  return new Promise((resolve, reject) => {
    server.on('error', (err) => {
      reject(err);
    });
    server.listen(port, '127.0.0.1', () => {
      resolve({ server, port });
    });
  });
}

function serveFile(req, res, filePath, explicitExt = null) {
  const stat = fs.statSync(filePath);
  const ext = explicitExt || path.extname(filePath).toLowerCase();
  const contentType = MIME_MAP[ext] || 'application/octet-stream';

  const rangeHeader = req.headers.range;
  if (rangeHeader) {
    const parts = rangeHeader.replace(/bytes=/, '').split('-');
    const start = parseInt(parts[0], 10);
    const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;

    if (start >= stat.size || end >= stat.size || start > end) {
      res.writeHead(416, {
        'Content-Range': `bytes */${stat.size}`,
      });
      res.end();
      return;
    }

    const chunksize = end - start + 1;
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${stat.size}`,
      'Accept-Ranges': 'bytes',
      'Content-Length': chunksize,
      'Content-Type': contentType,
    });
    fs.createReadStream(filePath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, {
      'Content-Length': stat.size,
      'Content-Type': contentType,
      'Accept-Ranges': 'bytes',
    });
    fs.createReadStream(filePath).pipe(res);
  }
}

// Standalone execution
if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const args = process.argv.slice(2);
  let port = 8999;
  const portIdx = args.indexOf('--port');
  if (portIdx !== -1 && args[portIdx + 1]) {
    port = parseInt(args[portIdx + 1], 10);
  }

  createHarnessServer({ port })
    .then(({ port: boundPort }) => {
      console.log(`[ASR.js Harness Server] Running at http://127.0.0.1:${boundPort}`);
      console.log(`  UI Harness: http://127.0.0.1:${boundPort}/harness.html`);
      console.log(`  Models API: http://127.0.0.1:${boundPort}/__api__/local-models`);
    })
    .catch((err) => {
      console.error('[ASR.js Harness Server] Failed to start:', err);
      process.exit(1);
    });
}
