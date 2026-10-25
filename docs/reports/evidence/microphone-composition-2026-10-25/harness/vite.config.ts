import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  publicDir: fileURLToPath(new URL('../../../demos/apps/realtime/public/', import.meta.url)),
  worker: { format: 'es' },
  optimizeDeps: { exclude: ['@asrjs/speech-recognition'] },
  server: {
    host: '127.0.0.1', port: 4341, strictPort: true,
    fs: { allow: [fileURLToPath(new URL('../../', import.meta.url)), fileURLToPath(new URL('../../../speech-recognition/', import.meta.url))] },
    headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' },
  },
});
