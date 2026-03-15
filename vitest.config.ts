import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: [
      { find: /^asr\.js$/, replacement: resolve(__dirname, 'src/index.ts') },
      { find: /^asr\.js\/builtins$/, replacement: resolve(__dirname, 'src/builtins.ts') },
      { find: /^asr\.js\/io$/, replacement: resolve(__dirname, 'src/io.ts') },
      { find: /^asr\.js\/io\/node$/, replacement: resolve(__dirname, 'src/io-node.ts') },
      { find: /^asr\.js\/inference$/, replacement: resolve(__dirname, 'src/inference.ts') },
      { find: /^asr\.js\/browser$/, replacement: resolve(__dirname, 'src/browser.ts') },
      { find: /^asr\.js\/realtime$/, replacement: resolve(__dirname, 'src/realtime.ts') },
      { find: /^asr\.js\/bench$/, replacement: resolve(__dirname, 'src/bench.ts') },
      { find: /^asr\.js\/datasets$/, replacement: resolve(__dirname, 'src/datasets.ts') },
      { find: /^asr\.js\/models\/(.+)$/, replacement: resolve(__dirname, 'src/models/$1.ts') },
      { find: /^asr\.js\/presets\/(.+)$/, replacement: resolve(__dirname, 'src/presets/$1.ts') }
    ]
  },
  test: {
    environment: 'node',
    include: [
      'tests/**/*.test.ts'
    ]
  }
});
