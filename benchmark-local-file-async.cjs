const { loadSplitGraphLocalModel } = require('./dist/models/whisper-seq2seq/local-file.js');

async function run() {
  const fixtureDir = './tests/fixtures/whisper-tiny-dummy';
  const start = performance.now();
  for (let i = 0; i < 10000; i++) {
    await loadSplitGraphLocalModel(fixtureDir);
  }
  const end = performance.now();
  console.log(`Time: ${end - start} ms`);
}

run();
