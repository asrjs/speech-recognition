/**
 * sync-to-hf.js
 * 
 * Syncs the main demo source code from examples/demo/ to the HF Spaces repo.
 * 
 * Since HF Spaces now builds from source (not dist), this script copies
 * the source files and adjusts paths for the standalone HF repo structure.
 * 
 * Usage:
 *   node scripts/sync-to-hf.js [--hf-repo-path <path>]
 * 
 * Default HF repo path: N:\github\ysdede\parakeet.js-demo
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PROJECT_ROOT = path.resolve(__dirname, '..');

// Parse args
const args = process.argv.slice(2);
const hfRepoPathIdx = args.indexOf('--hf-repo-path');
const HF_REPO_PATH = hfRepoPathIdx >= 0 && args[hfRepoPathIdx + 1]
    ? path.resolve(args[hfRepoPathIdx + 1])
    : path.resolve(PROJECT_ROOT, '../../../parakeet.js-demo');

const DRY_RUN = args.includes('--dry-run');

console.log(`📦 Syncing demo to HF Spaces repo`);
console.log(`   Source: ${PROJECT_ROOT}`);
console.log(`   Target: ${HF_REPO_PATH}`);
if (DRY_RUN) console.log(`   ⚠️  DRY RUN — no files will be written`);

// Verify paths exist
if (!fs.existsSync(PROJECT_ROOT)) {
    console.error(`❌ Source project not found: ${PROJECT_ROOT}`);
    process.exit(1);
}
if (!fs.existsSync(HF_REPO_PATH)) {
    console.error(`❌ HF repo not found: ${HF_REPO_PATH}`);
    console.error(`   Clone it first: git clone https://huggingface.co/spaces/ysdede/parakeet.js-demo`);
    process.exit(1);
}

function copyFile(src, dest, transform) {
    const relative = path.relative(PROJECT_ROOT, src);
    if (!fs.existsSync(src)) {
        console.warn(`   ⚠️  Skip (not found): ${relative}`);
        return;
    }
    if (DRY_RUN) {
        console.log(`   📄 Would copy: ${relative}`);
        return;
    }
    const dir = path.dirname(dest);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }
    if (transform) {
        let content = fs.readFileSync(src, 'utf8');
        content = transform(content);
        fs.writeFileSync(dest, content);
    } else {
        fs.copyFileSync(src, dest);
    }
    console.log(`   ✅ ${relative}`);
}

function copyDir(src, dest) {
    if (!fs.existsSync(src)) {
        console.warn(`   ⚠️  Skip dir (not found): ${src}`);
        return;
    }
    if (!DRY_RUN && !fs.existsSync(dest)) {
        fs.mkdirSync(dest, { recursive: true });
    }
    const entries = fs.readdirSync(src, { withFileTypes: true });
    for (const entry of entries) {
        const srcPath = path.join(src, entry.name);
        const destPath = path.join(dest, entry.name);
        if (entry.isDirectory()) {
            copyDir(srcPath, destPath);
        } else {
            copyFile(srcPath, destPath);
        }
    }
}

// No import path transform needed — both repos use './shared/' imports now.

console.log('\n📂 Syncing source files...');

// --- Source files ---
copyFile(
    path.join(PROJECT_ROOT, 'src/App.jsx'),
    path.join(HF_REPO_PATH, 'src/App.jsx'),
);
copyFile(
    path.join(PROJECT_ROOT, 'src/App.css'),
    path.join(HF_REPO_PATH, 'src/App.css'),
);
copyFile(
    path.join(PROJECT_ROOT, 'src/main.jsx'),
    path.join(HF_REPO_PATH, 'src/main.jsx'),
);

// --- Utils ---
copyDir(
    path.join(PROJECT_ROOT, 'src/utils'),
    path.join(HF_REPO_PATH, 'src/utils'),
);

// --- Shared modules (src/shared/ → src/shared/) ---
console.log('\n📂 Syncing shared modules...');
copyDir(
    path.join(PROJECT_ROOT, 'src/shared'),
    path.join(HF_REPO_PATH, 'src/shared'),
);

// --- Config files (only ones that should be identical) ---
console.log('\n📂 Syncing config files...');
copyFile(
    path.join(PROJECT_ROOT, 'tailwind.config.js'),
    path.join(HF_REPO_PATH, 'tailwind.config.js'),
);
copyFile(
    path.join(PROJECT_ROOT, 'postcss.config.js'),
    path.join(HF_REPO_PATH, 'postcss.config.js'),
);

// --- Public assets (skip coi-serviceworker.js — HF has real headers) ---
console.log('\n📂 Syncing public assets...');
copyFile(
    path.join(PROJECT_ROOT, 'public/favicon.ico'),
    path.join(HF_REPO_PATH, 'public/favicon.ico'),
);
copyDir(
    path.join(PROJECT_ROOT, 'public/assets'),
    path.join(HF_REPO_PATH, 'public/assets'),
);

// --- Space template ---
console.log('\n📂 Syncing space template...');
copyDir(
    path.join(PROJECT_ROOT, 'space_template'),
    path.join(HF_REPO_PATH, 'space_template'),
);

// --- Files NOT synced (intentionally different per platform) ---
console.log('\n📋 Files NOT synced (platform-specific):');
console.log('   • README.md          — HF version has YAML frontmatter');
console.log('   • index.html         — HF version omits coi-serviceworker.js');
console.log('   • vite.config.js     — HF version is NPM-only (no local-source aliasing)');
console.log('   • package.json       — HF version has no cross-env/dev:local scripts');
console.log('   • coi-serviceworker.js — Not needed on HF (real CORS headers)');

console.log('\n✨ Sync complete!');
console.log('   Next steps:');
console.log(`   1. cd ${HF_REPO_PATH}`);
console.log('   2. Review changes: git diff');
console.log('   3. Commit: git add -A && git commit -m "Sync from main demo"');
console.log('   4. Push: git push');
