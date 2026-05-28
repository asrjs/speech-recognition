import assert from 'node:assert/strict';
import {
  partitionWordsIntoSentences,
  transcriptToSrt,
  transcriptToVtt,
} from '../../dist/index.js';

const words = [
  { index: 0, text: 'The', startTime: 0, endTime: 0.12 },
  { index: 1, text: 'boy', startTime: 0.12, endTime: 0.35 },
  { index: 2, text: 'rose.', startTime: 0.35, endTime: 0.72 },
  { index: 3, text: 'Again', startTime: 3.9, endTime: 4.2 },
  { index: 4, text: 'now.', startTime: 4.2, endTime: 4.8 },
];
const sentences = partitionWordsIntoSentences(words);
const transcript = {
  text: words.map((word) => word.text).join(' '),
  warnings: [],
  meta: {
    detailLevel: 'sentences+words',
    isFinal: true,
    wordCount: words.length,
    sentenceCount: sentences.length,
  },
  sentences,
  words,
};

assert.equal(sentences.length, 2);
assert.match(transcriptToSrt(transcript), /00:00:00,000 --> 00:00:00,720/);
assert.match(transcriptToVtt(transcript, { source: 'words' }), /^WEBVTT\n\n/);
console.log('offline output smoke passed');
