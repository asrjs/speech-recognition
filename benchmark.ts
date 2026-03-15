import { MedAsrJsPreprocessor } from './src/models/lasr-ctc/mel.js';

const preprocessor = new MedAsrJsPreprocessor({ nMels: 80 });
const audioChunks = [];
for (let i = 0; i < 1000; i++) {
  const chunk = new Float32Array(16000);
  for (let j = 0; j < 16000; j++) {
    chunk[j] = Math.random() * 2 - 1;
  }
  audioChunks.push(chunk);
}

console.log("Starting benchmark...");
const start = performance.now();
for (const chunk of audioChunks) {
  preprocessor.process(chunk);
}
const end = performance.now();
console.log(`Elapsed time: ${end - start} ms`);
