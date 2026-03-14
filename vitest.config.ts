import { resolve } from 'node:path';
import { defineConfig } from 'vitest/config';

export default defineConfig({
  resolve: {
    alias: {
      'asr.js': resolve(__dirname, 'src/index.ts')
    }
  },
  test: {
    environment: 'node',
    include: [
      'tests/**/*.test.ts'
    ]
  }
});
