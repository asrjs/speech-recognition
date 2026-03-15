# Ground Truth for Merger Training and Genetic Agents

Use **utterances and words from the full JSON** as ground truth for merger evaluation and for training genetic/agent-based search for the best merging technique. **Do not depend on VTT** for that pipeline.

## Why skip VTT for merger ground truth?

VTT export (e.g. from `subtitle_export.write_vtt`) builds cues from segment text by applying sentence splitting (e.g. pysbd). That can introduce:

- **Fragment cues** – One word like "But" in its own cue with bad timestamps (e.g. same start/end).
- **Wrong boundaries** – A single utterance split into several cues because of sentence-boundary detection.
- **Cue-level noise** – Extra or missing cues relative to your real segments.

Those problems live in the **VTT export step**, not in your transcript. Your long-file JSON already has the correct **segments** (utterances) and **words** with timestamps.

## Use JSON as ground truth

For merger simulators and genetic agents:

1. **Reference** = `full_text` and segment texts from the same JSON (utterances). No VTT file needed.
2. **Word-level reference** = top-level `words` (and optionally words inside each segment) for finer metrics if you add them.
3. **Chunks for replay** = `asr_results` or derived from `segments` + `words`, or from `words` via `--chunk-by-window`.

**realtime_merger_sim** already does this when you omit `--ref-vtt`:

- It uses `reference_from_json(data)`: reference text = `full_text`, reference sentences = segment `text` (one per utterance).
- So running e.g. `python scripts/realtime_merger_sim.py simulation_ref_full.json --mergers python` uses JSON-only reference and avoids VTT entirely.

**Optuna / strategy_sweep** also take the full JSON as input; reference and chunks come from that file. No VTT is required for finding the best merger or window/step.

## Pipeline for genetic agents

1. Produce **one full JSON** per audio (e.g. `long_file_transcribe.py` or `long_file_transcribe_full.py` or `realtime_transcribe_sim.py`) with `full_text`, `words`, `segments`, and optionally `asr_results`.
2. Use that JSON as the **single source of truth** for:
   - Reference text and reference sentences (utterances).
   - Chunks to feed into the merger (from `asr_results` or segments+words or `--chunk-by-window`).
3. Run merger trials (realtime_merger_sim, Optuna, strategy_sweep) and compare merged output to the reference from the same JSON (e.g. WER, sentence recall, LCS recall).
4. **Optionally** export VTT for humans (e.g. subtitles); fix or improve VTT export separately. Do not use VTT as the ground truth for merger optimization.

## Summary

| Purpose                         | Use                         | Avoid for this purpose   |
|---------------------------------|-----------------------------|---------------------------|
| Merger evaluation / optimization| JSON: segments + words      | VTT as reference          |
| Genetic/agent merger search     | JSON: full_text + segments  | VTT as reference          |
| Human-readable subtitles        | VTT (optional, fix separately) | Relying on VTT for metrics |
