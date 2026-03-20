export interface SpeechModelLocalFileHandleLike {
  readonly kind?: 'file';
  getFile(): Promise<File | Blob>;
}

export interface SpeechModelLocalDirectoryHandleLike {
  readonly kind?: 'directory';
  readonly name?: string;
  entries(): AsyncIterable<
    [string, SpeechModelLocalFileHandleLike | SpeechModelLocalDirectoryHandleLike]
  >;
}

export interface SpeechModelLocalEntry {
  readonly path: string;
  readonly basename: string;
  readonly file?: File | Blob;
  readonly handle?: SpeechModelLocalFileHandleLike;
}
