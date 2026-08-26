import { describe, expect, it } from 'vitest';
import {
  createBrowserCompactStatsRenderer,
  resolveBrowserCompactStatsSnapshot,
  type BrowserCompactStatsSnapshot,
} from '../src/runtime/browser-compact-stats.js';

class MockHTMLElement {
  attributes = new Map<string, string>();
  children: MockHTMLElement[] = [];
  style = { cssText: '' };
  textContent = '';
  innerHTML = '';
  tagName = 'DIV';

  constructor(tagName = 'DIV') {
    this.tagName = tagName.toUpperCase();
  }

  setAttribute(key: string, value: string) {
    this.attributes.set(key, value);
  }

  getAttribute(key: string) {
    return this.attributes.get(key) ?? null;
  }

  appendChild(child: MockHTMLElement) {
    this.children.push(child);
    return child;
  }

  replaceChildren(...children: MockHTMLElement[]) {
    this.children = children;
  }

  querySelector(selector: string) {
    for (const child of this.children) {
      if (child.tagName.toLowerCase() === selector.toLowerCase()) {
        return child;
      }
      const found = child.querySelector(selector);
      if (found) return found;
    }
    return null;
  }

  get textContentAll(): string {
    let result = this.textContent;
    for (const child of this.children) {
      result += child.textContentAll;
    }
    return result;
  }
}

function createMockDocument() {
  return {
    createElement(tagName: string) {
      return new MockHTMLElement(tagName);
    },
  };
}

describe('createBrowserCompactStatsRenderer', () => {
  it('creates container attributes and renders DOM elements safely', () => {
    const originalDoc = (globalThis as any).document;
    (globalThis as any).document = createMockDocument();
    try {
      const container = document.createElement('div') as any;
      const renderer = createBrowserCompactStatsRenderer(container);

    expect(container.getAttribute('role')).toBe('status');
    expect(container.getAttribute('aria-live')).toBe('polite');
    expect(container.getAttribute('aria-label')).toBe('Live detector summary');

    const snapshot: BrowserCompactStatsSnapshot = {
      speaking: true,
      currentSnr: 15.5,
      snrThreshold: 10.0,
      minSnrThreshold: 5.0,
      signalDbfs: -20.5,
      averageDbfs: -30.0,
      gateDbfs: -25.0,
      noiseDbfs: -45.0,
      targetRate: 16000,
      visibleSeconds: 5.0,
      recentSegments: 3,
      recentRejected: 1,
    };

    renderer.update(snapshot);

    expect(container.children.length).toBe(1);
    const fullText = container.textContentAll;
    expect(fullText).toContain('15.50');
    expect(fullText).toContain('Sig-20.5');
    expect(fullText).toContain('Avg-30.0');
    expect(fullText).toContain('Gate-25.0');
    expect(fullText).toContain('Noise-45.0');
    expect(fullText).toContain('Segs3/1');
    expect(fullText).toContain('Audio16000Hz');
    expect(fullText).toContain('Buf5.0s');
    expect(fullText).toContain('SNR15.50');
    expect(fullText).toContain('Thr10.00');

      renderer.dispose();
      expect(container.innerHTML).toBe('');
    } finally {
      (globalThis as any).document = originalDoc;
    }
  });

  it('prevents XSS injection when resolution logic processes unexpected inputs', () => {
    const originalDoc = (globalThis as any).document;
    (globalThis as any).document = createMockDocument();
    try {
      const container = document.createElement('div') as any;
      const renderer = createBrowserCompactStatsRenderer(container);

      // Simulate potential malicious or unusual data fed into snapshot calculation
      const maliciousSource = {
        recentDecisions: [{ message: '<script>alert("xss")</script> Segment rejected' }],
        recentSegments: [1, 2, 3],
      };

      const snapshot = resolveBrowserCompactStatsSnapshot(maliciousSource as any, null, null);
      renderer.update(snapshot);

      // Verify script tags are not executed or parsed into real script elements
      expect(container.querySelector('script')).toBeNull();
      // Segs label should safely render formatted text "Segs3/1"
      expect(container.textContentAll).toContain('Segs3/1');
    } finally {
      (globalThis as any).document = originalDoc;
    }
  });
});
