export interface ModelClassification {
  readonly ecosystem: string;
  readonly processor?: string;
  readonly encoder?: string;
  readonly decoder?: string;
  readonly topology?: string;
  readonly family?: string;
  readonly task: string;
}
