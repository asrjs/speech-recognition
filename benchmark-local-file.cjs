const { loadSplitGraphLocalModel } = require('./dist/models/whisper-seq2seq/local-file.js');
const fs = require('fs');

async function run() {
  const fixtureDir = './tests/fixtures/whisper-tiny-dummy';

  if (!fs.existsSync(fixtureDir)) {
    console.log("No fixture dir to benchmark against, skipping benchmarking.");
    return;
  }

  const start = performance.now();
  for (let i = 0; i < 10000; i++) {
    loadSplitGraphLocalModel(fixtureDir);
  }
  const end = performance.now();

  console.log(`Time: ${end - start} ms`);
}

run();
