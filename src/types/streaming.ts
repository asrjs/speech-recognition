import type { AudioInputLike } from './audio.js';
import type { TranscriptDetailLevel, PartialTranscript } from './transcript.js';

export interface VoiceActivityEvent {
  readonly isSpeech: boolean;
  readonly speechProbability: number;
  readonly startTimeSeconds?: number;
  readonly endTimeSeconds?: number;
}

export interface VoiceActivityDetector {
  analyze(input: AudioInputLike): Promise<VoiceActivityEvent> | VoiceActivityEvent;
  reset(): Promise<void> | void;
}

export interface StreamingSessionOptions {
  readonly detail?: TranscriptDetailLevel;
  readonly maxWindowMs?: number;
  readonly overlapMs?: number;
  readonly emitPartials?: boolean;
  readonly minFinalSilenceMs?: number;
  readonly vad?: VoiceActivityDetector;
}

export interface StreamingTranscriberState {
  readonly revision: number;
  readonly bufferedDurationSeconds: number;
  readonly committedText: string;
  readonly previewText: string;
  readonly isFinalized: boolean;
}

export interface StreamingTranscriber {
  pushAudio(input: AudioInputLike): Promise<PartialTranscript>;
  flush(): Promise<PartialTranscript>;
  finalize(): Promise<PartialTranscript>;
  reset(): Promise<void> | void;
  getState(): StreamingTranscriberState;
}

