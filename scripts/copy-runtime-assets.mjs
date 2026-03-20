import { cpSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

const copies = [
  {
    from: resolve('src/runtime/assets/ten-vad'),
    to: resolve('dist/runtime/assets/ten-vad'),
  },
];

for (const entry of copies) {
  if (!existsSync(entry.from)) {
    continue;
  }
  mkdirSync(dirname(entry.to), { recursive: true });
  cpSync(entry.from, entry.to, { recursive: true, force: true });
}
