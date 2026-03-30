export class ConfigAliasConflictError extends Error {
  readonly code = 'CONFIG_ALIAS_CONFLICT';

  constructor(message: string) {
    super(message);
    this.name = 'ConfigAliasConflictError';
  }
}

export class FireRedRuntimeError extends Error {
  readonly code = 'FIRERED_RUNTIME_ERROR';

  constructor(message: string) {
    super(message);
    this.name = 'FireRedRuntimeError';
  }
}
