import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import zlib from 'node:zlib';

const PROJECT_ROOT = process.cwd();

const TARGETS = {
  canary: {
    description: 'Large Canary NeMo reference JSON used by parity/debugging scripts.',
    files: ['tools/data/results/canary/canary-180m-flash-reference.json'],
  },
  medasr: {
    description: 'Large MedASR upstream parity tensors used by copied debugging tests.',
    files: [
      'tools/model-debugging/reference/medasrjs/upstream-tests/reference_medasr/features.json',
      'tools/model-debugging/reference/medasrjs/upstream-tests/reference_medasr/logits.json',
    ],
  },
  parakeet: {
    description: 'Large Parakeet realtime RNNT reference JSON used by parity/debugging scripts.',
    files: ['tools/data/results/parakeet/parakeet-realtime-eou-120m-v1-reference.json'],
  },
};

function parseArgs(argv) {
  const options = {
    command: 'status',
    targets: [],
    deleteOriginals: false,
    force: false,
  };

  let commandSet = false;
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (!commandSet && !arg.startsWith('--')) {
      options.command = arg;
      commandSet = true;
      continue;
    }

    if (arg === '--target') {
      options.targets.push(argv[++index]);
      continue;
    }
    if (arg === '--delete-originals') {
      options.deleteOriginals = true;
      continue;
    }
    if (arg === '--force') {
      options.force = true;
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return options;
}

function resolveTargets(selectedTargets) {
  const targetNames = selectedTargets.length > 0 ? selectedTargets : Object.keys(TARGETS);
  return targetNames.map((name) => {
    const target = TARGETS[name];
    if (!target) {
      throw new Error(`Unknown target "${name}". Available targets: ${Object.keys(TARGETS).join(', ')}`);
    }
    return {
      name,
      description: target.description,
      files: target.files.map((file) => path.resolve(PROJECT_ROOT, file)),
    };
  });
}

function sha256(buffer) {
  return crypto.createHash('sha256').update(buffer).digest('hex');
}

function gzipPathFor(filePath) {
  return `${filePath}.gz`;
}

function readManifest(manifestPath) {
  if (!fs.existsSync(manifestPath)) {
    return null;
  }
  return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
}

function writeManifest(manifestPath, payload) {
  fs.writeFileSync(manifestPath, `${JSON.stringify(payload, null, 2)}\n`, 'utf8');
}

function uniqueManifestPaths(targets) {
  const manifests = new Map();
  for (const target of targets) {
    for (const filePath of target.files) {
      const dir = path.dirname(filePath);
      const manifestPath = path.join(dir, 'reference-artifacts.manifest.json');
      if (!manifests.has(manifestPath)) {
        manifests.set(manifestPath, []);
      }
      manifests.get(manifestPath).push(filePath);
    }
  }
  return manifests;
}

function packTargets(targets, options) {
  const manifests = uniqueManifestPaths(targets);
  for (const [manifestPath, filePaths] of manifests.entries()) {
    const prior = readManifest(manifestPath);
    const entries = [];
    for (const filePath of filePaths) {
      if (!fs.existsSync(filePath)) {
        if (options.force && fs.existsSync(gzipPathFor(filePath))) {
          continue;
        }
        throw new Error(`Cannot pack missing file: ${filePath}`);
      }
      const raw = fs.readFileSync(filePath);
      const zipped = zlib.gzipSync(raw, { level: 9 });
      const gzipPath = gzipPathFor(filePath);
      fs.writeFileSync(gzipPath, zipped);
      entries.push({
        file: path.basename(filePath),
        gzipFile: path.basename(gzipPath),
        sizeBytes: raw.byteLength,
        gzipBytes: zipped.byteLength,
        sha256: sha256(raw),
      });
      if (options.deleteOriginals) {
        fs.rmSync(filePath);
      }
    }

    const mergedEntries = new Map();
    for (const entry of prior?.entries ?? []) {
      mergedEntries.set(entry.file, entry);
    }
    for (const entry of entries) {
      mergedEntries.set(entry.file, entry);
    }

    writeManifest(manifestPath, {
      version: 1,
      updatedAt: new Date().toISOString(),
      entries: [...mergedEntries.values()].sort((left, right) => left.file.localeCompare(right.file)),
    });

    console.log(`Packed ${entries.length} artifact(s) into ${path.relative(PROJECT_ROOT, path.dirname(manifestPath))}`);
  }
}

function unpackTargets(targets, options) {
  const manifests = uniqueManifestPaths(targets);
  for (const [manifestPath, filePaths] of manifests.entries()) {
    const manifest = readManifest(manifestPath);
    if (!manifest) {
      throw new Error(`Manifest not found: ${manifestPath}`);
    }

    let unpacked = 0;
    for (const filePath of filePaths) {
      const gzipPath = gzipPathFor(filePath);
      if (!fs.existsSync(gzipPath)) {
        throw new Error(`Compressed artifact not found: ${gzipPath}`);
      }
      if (fs.existsSync(filePath) && !options.force) {
        continue;
      }
      const raw = zlib.gunzipSync(fs.readFileSync(gzipPath));
      const entry = manifest.entries.find((candidate) => candidate.file === path.basename(filePath));
      if (entry && sha256(raw) !== entry.sha256) {
        throw new Error(`Checksum mismatch while unpacking ${filePath}`);
      }
      fs.writeFileSync(filePath, raw);
      unpacked += 1;
    }

    console.log(`Unpacked ${unpacked} artifact(s) into ${path.relative(PROJECT_ROOT, path.dirname(manifestPath))}`);
  }
}

function statusTargets(targets) {
  for (const target of targets) {
    console.log(`${target.name}: ${target.description}`);
    for (const filePath of target.files) {
      const rawExists = fs.existsSync(filePath);
      const gzipPath = gzipPathFor(filePath);
      const gzipExists = fs.existsSync(gzipPath);
      const rawSize = rawExists ? fs.statSync(filePath).size : 0;
      const gzipSize = gzipExists ? fs.statSync(gzipPath).size : 0;
      console.log(
        `  - ${path.relative(PROJECT_ROOT, filePath)} | raw=${rawExists ? rawSize : 'missing'} | gzip=${gzipExists ? gzipSize : 'missing'}`,
      );
    }
  }
}

function printHelp() {
  console.log(`Usage:
  node tools/model-debugging/scripts/node-reference-artifacts.mjs status [--target canary] [--target medasr] [--target parakeet]
  node tools/model-debugging/scripts/node-reference-artifacts.mjs pack --target canary --delete-originals
  node tools/model-debugging/scripts/node-reference-artifacts.mjs unpack --target medasr

Targets:
${Object.entries(TARGETS)
  .map(([name, target]) => `  - ${name}: ${target.description}`)
  .join('\n')}`);
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.command === 'help' || options.command === '--help' || options.command === '-h') {
    printHelp();
    return;
  }

  const targets = resolveTargets(options.targets);
  if (options.command === 'pack') {
    packTargets(targets, options);
    return;
  }
  if (options.command === 'unpack') {
    unpackTargets(targets, options);
    return;
  }
  if (options.command === 'status') {
    statusTargets(targets);
    return;
  }

  throw new Error(`Unknown command "${options.command}". Use status, pack, unpack, or help.`);
}

try {
  main();
} catch (error) {
  console.error('[node-reference-artifacts] failed:', error);
  process.exitCode = 1;
}
