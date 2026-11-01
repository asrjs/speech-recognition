import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SPEECH_RECOGNITION_DIR = path.resolve(__dirname, '../..');
const DOCS_DIR = path.resolve(SPEECH_RECOGNITION_DIR, 'docs');

function getAllMarkdownFiles(dir) {
  let results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    if (stat.isDirectory()) {
      results = results.concat(getAllMarkdownFiles(fullPath));
    } else if (file.endsWith('.md')) {
      results.push(fullPath);
    }
  }
  return results;
}

test('all relative markdown links in speech-recognition/docs resolve to existing files', () => {
  assert.ok(fs.existsSync(DOCS_DIR), `Docs directory must exist: ${DOCS_DIR}`);

  const files = getAllMarkdownFiles(DOCS_DIR);
  assert.ok(files.length > 0, 'Must find markdown documentation files');

  const inlineLinkRegex = /\[([^\]]*)\]\(([^)]+)\)/g;
  const refLinkDefRegex = /^\s*\[(\^?[^\]]+)\]:\s*(\S+)/gm;
  const htmlLinkRegex = /<a\s+[^>]*href=["']([^"']+)["']/gi;

  let totalRelativeLinksChecked = 0;
  const brokenLinks = [];

  for (const filePath of files) {
    const content = fs.readFileSync(filePath, 'utf8');
    const relSourcePath = path.relative(SPEECH_RECOGNITION_DIR, filePath);
    const foundUrls = [];

    let match;
    while ((match = inlineLinkRegex.exec(content)) !== null) {
      let rawUrl = match[2].trim();
      if (rawUrl.includes(' ')) {
        rawUrl = rawUrl.split(/\s+/)[0];
      }
      rawUrl = rawUrl.replace(/^<|>$/g, '');
      const lineNumber = content.substring(0, match.index).split('\n').length;
      foundUrls.push({ url: rawUrl, text: match[1], line: lineNumber });
    }

    while ((match = refLinkDefRegex.exec(content)) !== null) {
      const label = match[1];
      if (label.startsWith('^')) {
        continue; // footnote definition
      }
      let rawUrl = match[2].trim();
      if (rawUrl.includes(' ')) {
        rawUrl = rawUrl.split(/\s+/)[0];
      }
      rawUrl = rawUrl.replace(/^<|>$/g, '');
      const lineNumber = content.substring(0, match.index).split('\n').length;
      foundUrls.push({ url: rawUrl, text: label, line: lineNumber });
    }

    while ((match = htmlLinkRegex.exec(content)) !== null) {
      let rawUrl = match[1].trim();
      const lineNumber = content.substring(0, match.index).split('\n').length;
      foundUrls.push({ url: rawUrl, text: '', line: lineNumber });
    }

    for (const item of foundUrls) {
      const url = item.url;
      // Skip web and non-file URI protocols
      if (/^(https?:|mailto:|ftp:|data:|javascript:|chrome:|sidecar:)/i.test(url)) {
        continue;
      }

      totalRelativeLinksChecked++;
      const [targetPath] = url.split('#');

      // Pure anchor inside current document
      if (!targetPath) {
        continue;
      }

      let resolvedTarget;
      if (targetPath.startsWith('/')) {
        resolvedTarget = path.resolve(SPEECH_RECOGNITION_DIR, '.' + targetPath);
      } else {
        resolvedTarget = path.resolve(path.dirname(filePath), targetPath);
      }

      if (!fs.existsSync(resolvedTarget)) {
        brokenLinks.push({
          sourceFile: relSourcePath,
          line: item.line,
          text: item.text,
          url: item.url,
          resolvedTarget: path.relative(SPEECH_RECOGNITION_DIR, resolvedTarget),
        });
      }
    }
  }

  console.log(
    `Docs link smoke test: checked ${totalRelativeLinksChecked} relative links across ${files.length} markdown files.`,
  );

  if (brokenLinks.length > 0) {
    const errorDetails = brokenLinks
      .map(
        (b) =>
          `  - ${b.sourceFile}:${b.line} [${b.text}](${b.url}) -> resolved: ${b.resolvedTarget}`,
      )
      .join('\n');
    assert.fail(`Found ${brokenLinks.length} broken relative links in docs:\n${errorDetails}`);
  }

  assert.equal(brokenLinks.length, 0, 'Zero broken relative links expected in docs');
});
