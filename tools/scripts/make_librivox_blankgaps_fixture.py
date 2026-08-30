"""Generate a blank-dominant synthetic TDT measurement fixture.

Derives tools/data/fixtures/audio/librivox-blankgaps-synthetic.wav from
librivox.org.wav by keeping the first 15 s of speech and inserting nine
3.2 s silence gaps between equal speech segments.

Provenance: SYNTHETIC clip for dispatch-mechanics measurement only. It is
NOT a benchmark-quality fixture and must never be used for accuracy or
RTFx claims against published corpora. The silence is pure zeros at the
source sample rate; no re-encoding or resampling is applied.
"""
from __future__ import annotations

import wave
from pathlib import Path

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "fixtures" / "audio"
SOURCE = FIXTURES / "librivox.org.wav"
TARGET = FIXTURES / "librivox-blankgaps-synthetic.wav"

SPEECH_SECONDS = 15.0
GAP_SECONDS = 3.2
SEGMENTS = 9


def main() -> None:
    with wave.open(str(SOURCE), "rb") as reader:
        channels = reader.getnchannels()
        width = reader.getsampwidth()
        rate = reader.getframerate()
        if channels != 1 or width != 2:
            raise SystemExit(f"expected mono 16-bit source, got ch={channels} width={width}")
        speech = reader.readframes(int(rate * SPEECH_SECONDS))

    frame_size = width * channels
    speech_frames = len(speech) // frame_size
    seg_frames = speech_frames // SEGMENTS
    gap_frames = int(rate * GAP_SECONDS)
    silence = b"\x00" * (gap_frames * frame_size)

    parts: list[bytes] = []
    for index in range(SEGMENTS):
        start = index * seg_frames
        end = (index + 1) * seg_frames if index < SEGMENTS - 1 else speech_frames
        parts.append(speech[start * frame_size : end * frame_size])
        if index < SEGMENTS - 1:
            parts.append(silence)

    payload = b"".join(parts)
    with wave.open(str(TARGET), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(payload)

    total_seconds = len(payload) / frame_size / rate
    print(f"wrote {TARGET.name}: {total_seconds:.2f}s, {SEGMENTS} gaps x {GAP_SECONDS}s")


if __name__ == "__main__":
    main()

