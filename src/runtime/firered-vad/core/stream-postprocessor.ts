import type { StreamVadFrameResult } from '../types.js';

enum VadState {
  SILENCE = 0,
  POSSIBLE_SPEECH = 1,
  SPEECH = 2,
  POSSIBLE_SILENCE = 3,
}

export interface StreamVadPostprocessorOptions {
  readonly smoothWindowSize: number;
  readonly speechThreshold: number;
  readonly padStartFrame: number;
  readonly minSpeechFrame: number;
  readonly maxSpeechFrame: number;
  readonly minSilenceFrame: number;
}

export class StreamVadPostprocessor {
  smoothWindowSize: number;
  speechThreshold: number;
  padStartFrame: number;
  minSpeechFrame: number;
  maxSpeechFrame: number;
  minSilenceFrame: number;

  private frameCnt = 0;
  private smoothWindow: number[] = [];
  private smoothWindowSum = 0;
  private state = VadState.SILENCE;
  private speechCnt = 0;
  private silenceCnt = 0;
  private hitMaxSpeech = false;
  private lastSpeechStartFrame = -1;
  private lastSpeechEndFrame = -1;

  constructor(options: StreamVadPostprocessorOptions) {
    this.smoothWindowSize = Math.max(1, options.smoothWindowSize);
    this.speechThreshold = options.speechThreshold;
    this.padStartFrame = Math.max(this.smoothWindowSize, options.padStartFrame);
    this.minSpeechFrame = options.minSpeechFrame;
    this.maxSpeechFrame = options.maxSpeechFrame;
    this.minSilenceFrame = options.minSilenceFrame;
  }

  reset(): void {
    this.frameCnt = 0;
    this.smoothWindow = [];
    this.smoothWindowSum = 0;
    this.state = VadState.SILENCE;
    this.speechCnt = 0;
    this.silenceCnt = 0;
    this.hitMaxSpeech = false;
    this.lastSpeechStartFrame = -1;
    this.lastSpeechEndFrame = -1;
  }

  processOneFrame(rawProb: number): StreamVadFrameResult {
    this.frameCnt += 1;
    const smoothedProb = this.smoothProb(rawProb);
    const isSpeech = Number(smoothedProb >= this.speechThreshold) === 1;

    const result = {
      frame_idx: this.frameCnt,
      frameIdx: this.frameCnt,
      is_speech: isSpeech,
      isSpeech,
      raw_prob: Math.round(rawProb * 1000) / 1000,
      rawProb: Math.round(rawProb * 1000) / 1000,
      smoothed_prob: Math.round(smoothedProb * 1000) / 1000,
      smoothedProb: Math.round(smoothedProb * 1000) / 1000,
      is_speech_start: false,
      isSpeechStart: false,
      is_speech_end: false,
      isSpeechEnd: false,
      speech_start_frame: -1,
      speechStartFrame: -1,
      speech_end_frame: -1,
      speechEndFrame: -1,
    };

    return this.stateTransition(isSpeech, result);
  }

  private smoothProb(prob: number): number {
    if (this.smoothWindowSize <= 1) {
      return prob;
    }
    this.smoothWindow.push(prob);
    this.smoothWindowSum += prob;
    if (this.smoothWindow.length > this.smoothWindowSize) {
      const left = this.smoothWindow.shift() ?? 0;
      this.smoothWindowSum -= left;
    }
    return this.smoothWindowSum / this.smoothWindow.length;
  }

  private stateTransition(
    isSpeech: boolean,
    result: StreamVadFrameResult,
  ): StreamVadFrameResult {
    const mutable = { ...result };
    if (this.hitMaxSpeech) {
      mutable.is_speech_start = true;
      mutable.isSpeechStart = true;
      mutable.speech_start_frame = this.frameCnt;
      mutable.speechStartFrame = this.frameCnt;
      this.lastSpeechStartFrame = mutable.speech_start_frame;
      this.hitMaxSpeech = false;
    }

    if (this.state === VadState.SILENCE) {
      if (isSpeech) {
        this.state = VadState.POSSIBLE_SPEECH;
        this.speechCnt += 1;
      } else {
        this.silenceCnt += 1;
        this.speechCnt = 0;
      }
    } else if (this.state === VadState.POSSIBLE_SPEECH) {
      if (isSpeech) {
        this.speechCnt += 1;
        if (this.speechCnt >= this.minSpeechFrame) {
          this.state = VadState.SPEECH;
          const start = Math.max(
            1,
            this.frameCnt - this.speechCnt + 1 - this.padStartFrame,
            this.lastSpeechEndFrame + 1,
          );
          mutable.is_speech_start = true;
          mutable.isSpeechStart = true;
          mutable.speech_start_frame = start;
          mutable.speechStartFrame = start;
          this.lastSpeechStartFrame = start;
          this.silenceCnt = 0;
        }
      } else {
        this.state = VadState.SILENCE;
        this.silenceCnt = 1;
        this.speechCnt = 0;
      }
    } else if (this.state === VadState.SPEECH) {
      this.speechCnt += 1;
      if (isSpeech) {
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          mutable.is_speech_end = true;
          mutable.isSpeechEnd = true;
          mutable.speech_end_frame = this.frameCnt;
          mutable.speechEndFrame = this.frameCnt;
          mutable.speech_start_frame = this.lastSpeechStartFrame;
          mutable.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechStartFrame = -1;
          this.lastSpeechEndFrame = mutable.speech_end_frame;
        }
      } else {
        this.state = VadState.POSSIBLE_SILENCE;
        this.silenceCnt += 1;
      }
    } else {
      this.speechCnt += 1;
      if (isSpeech) {
        this.state = VadState.SPEECH;
        this.silenceCnt = 0;
        if (this.speechCnt >= this.maxSpeechFrame) {
          this.hitMaxSpeech = true;
          this.speechCnt = 0;
          mutable.is_speech_end = true;
          mutable.isSpeechEnd = true;
          mutable.speech_end_frame = this.frameCnt;
          mutable.speechEndFrame = this.frameCnt;
          mutable.speech_start_frame = this.lastSpeechStartFrame;
          mutable.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechStartFrame = -1;
          this.lastSpeechEndFrame = mutable.speech_end_frame;
        }
      } else {
        this.silenceCnt += 1;
        if (this.silenceCnt >= this.minSilenceFrame) {
          this.state = VadState.SILENCE;
          mutable.is_speech_end = true;
          mutable.isSpeechEnd = true;
          mutable.speech_end_frame = this.frameCnt;
          mutable.speechEndFrame = this.frameCnt;
          mutable.speech_start_frame = this.lastSpeechStartFrame;
          mutable.speechStartFrame = this.lastSpeechStartFrame;
          this.lastSpeechEndFrame = mutable.speech_end_frame;
          this.lastSpeechStartFrame = -1;
          this.speechCnt = 0;
        }
      }
    }

    return mutable;
  }
}
