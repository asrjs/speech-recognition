import type { AssetRequest, ResolvedAssetHandle } from '../types/index.js';
import { importNodeModule } from './node.js';

export class NodePathAssetHandle implements ResolvedAssetHandle {
  constructor(
    readonly request: AssetRequest,
    private readonly path: string,
  ) {}

  get contentType(): string | undefined {
    return this.request.contentType;
  }

  async *openStream(): AsyncIterable<Uint8Array> {
    const fs = await this.getFs();
    const stream = fs.createReadStream(this.path);
    for await (const chunk of stream) {
      yield chunk instanceof Uint8Array ? chunk : new Uint8Array(chunk);
    }
  }

  async readBytes(): Promise<Uint8Array> {
    const fs = await this.getFsPromises();
    const bytes = await fs.readFile(this.path);
    return new Uint8Array(bytes);
  }

  async readText(): Promise<string> {
    const fs = await this.getFsPromises();
    return fs.readFile(this.path, 'utf8');
  }

  async readJson<T>(): Promise<T> {
    return JSON.parse(await this.readText()) as T;
  }

  async getLocator(target: 'url' | 'path'): Promise<string | null> {
    if (target === 'path') {
      return this.path;
    }

    const { pathToFileURL } = await this.getUrl();
    return pathToFileURL(this.path).href;
  }

  private async getFs(): Promise<typeof import('node:fs')> {
    return importNodeModule<typeof import('node:fs')>('node:fs');
  }

  private async getFsPromises(): Promise<typeof import('node:fs/promises')> {
    return importNodeModule<typeof import('node:fs/promises')>('node:fs/promises');
  }

  private async getUrl(): Promise<typeof import('node:url')> {
    return importNodeModule<typeof import('node:url')>('node:url');
  }

  dispose(): void {
    return undefined;
  }
}
