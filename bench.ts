interface FrameAlignedToken {
  readonly id: number;
  readonly frameIndex: number;
  readonly absTime: number;
  readonly logProb: number;
  readonly text: string;
  readonly vignetteWeight?: number;
}

const timeTolerance = 0.2;

function findAnchorsOld(pendingTokens: FrameAlignedToken[], overlapTokens: readonly FrameAlignedToken[]): FrameAlignedToken[] {
  const anchors: FrameAlignedToken[] = [];
  for (const newToken of overlapTokens) {
    for (const pendingToken of pendingTokens) {
      if (
        newToken.id === pendingToken.id &&
        Math.abs(newToken.absTime - pendingToken.absTime) < timeTolerance
      ) {
        anchors.push(newToken);
        break;
      }
    }
  }

  return anchors.sort((left, right) => left.absTime - right.absTime);
}

function findAnchorsNew(pendingTokens: FrameAlignedToken[], overlapTokens: readonly FrameAlignedToken[]): FrameAlignedToken[] {
  const anchors: FrameAlignedToken[] = [];
  const pendingById = new Map<number, FrameAlignedToken[]>();
  for (const token of pendingTokens) {
    let arr = pendingById.get(token.id);
    if (!arr) {
      arr = [];
      pendingById.set(token.id, arr);
    }
    arr.push(token);
  }

  for (const newToken of overlapTokens) {
    const pendingMatchTokens = pendingById.get(newToken.id);
    if (pendingMatchTokens) {
      for (const pendingToken of pendingMatchTokens) {
        if (Math.abs(newToken.absTime - pendingToken.absTime) < timeTolerance) {
          anchors.push(newToken);
          break;
        }
      }
    }
  }

  return anchors.sort((left, right) => left.absTime - right.absTime);
}

const pendingTokens: FrameAlignedToken[] = [];
const overlapTokens: FrameAlignedToken[] = [];

for (let i = 0; i < 5000; i++) {
  pendingTokens.push({
    id: i % 100,
    frameIndex: i,
    absTime: i * 0.08,
    logProb: -0.1,
    text: 'a',
  });

  if (i % 2 === 0) {
    overlapTokens.push({
      id: i % 100,
      frameIndex: i + 0.1, // slightly different but within tolerance
      absTime: i * 0.08 + 0.1,
      logProb: -0.1,
      text: 'a',
    });
  }
}

// Warmup
for (let i = 0; i < 10; i++) {
  findAnchorsOld(pendingTokens, overlapTokens);
  findAnchorsNew(pendingTokens, overlapTokens);
}

const startOld = performance.now();
for (let i = 0; i < 100; i++) {
  findAnchorsOld(pendingTokens, overlapTokens);
}
const endOld = performance.now();

const startNew = performance.now();
for (let i = 0; i < 100; i++) {
  findAnchorsNew(pendingTokens, overlapTokens);
}
const endNew = performance.now();

console.log(`Old: ${endOld - startOld}ms`);
console.log(`New: ${endNew - startNew}ms`);
console.log(`Improvement: ${((endOld - startOld) / (endNew - startNew)).toFixed(2)}x`);
