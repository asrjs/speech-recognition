import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  MODELS,
  LANGUAGE_NAMES,
  ParakeetModel,
  benchmarkRunRecordsToCsv,
  decodeAudioSourceToMonoPcm,
  fetchDatasetSplits,
  fetchRandomRows,
  fetchSequentialRows,
  getConfigsAndSplits,
  getParakeetModel,
  loadModelWithFallback,
  normalizeDatasetRow,
  summarizeNumericSeries,
  textSimilarity,
} from 'asr.js';
import './App.css';

const SETTINGS_KEY = 'asrjs.benchmark-demo.settings.v1';
const VERSION = typeof __ASRJS_VERSION__ !== 'undefined' ? __ASRJS_VERSION__ : 'unknown';
const SOURCE = typeof __ASRJS_SOURCE__ !== 'undefined' ? __ASRJS_SOURCE__ : 'dev';
const BACKENDS = ['webgpu-hybrid', 'webgpu', 'wasm'];
const QUANTS = ['fp16', 'int8', 'fp32'];
const DATASET_SUGGESTIONS = [
  'ysdede/parrot-radiology-asr-en',
  'mozilla-foundation/common_voice_17_0',
  'MLCommons/peoples_speech',
  'facebook/multilingual_librispeech',
];

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}');
  } catch {
    return {};
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage failures in the demo.
  }
}

function mean(values) {
  const valid = values.filter(Number.isFinite);
  if (valid.length === 0) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function fmtMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : '-';
}

function fmtPct(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-';
}

function fmtNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : '-';
}

function downloadText(filename, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function summarizeRuns(runs) {
  const okRuns = runs.filter((run) => !run.error);
  const total = summarizeNumericSeries(okRuns.map((run) => run.metrics?.total_ms).filter(Number.isFinite));
  const encode = summarizeNumericSeries(okRuns.map((run) => run.metrics?.encode_ms).filter(Number.isFinite));
  const decode = summarizeNumericSeries(okRuns.map((run) => run.metrics?.decode_ms).filter(Number.isFinite));
  const similarity = summarizeNumericSeries(okRuns.map((run) => run.similarityToFirst).filter(Number.isFinite));
  const exactMatches = okRuns.filter((run) => run.exactMatchToFirst === true).length;
  const rtf = summarizeNumericSeries(okRuns.map((run) => run.metrics?.rtf).filter(Number.isFinite));
  return {
    runCount: runs.length,
    okCount: okRuns.length,
    errorCount: runs.length - okRuns.length,
    total,
    encode,
    decode,
    similarity,
    exactRate: okRuns.length > 0 ? exactMatches / okRuns.length : null,
    rtf,
  };
}

function collectConfigMap(items) {
  const grouped = getConfigsAndSplits(items);
  const configs = grouped.size > 0 ? [...grouped.keys()] : ['default'];
  const configToSplits = {};
  for (const [config, splits] of grouped.entries()) {
    configToSplits[config] = [...splits];
  }
  return {
    configs,
    configToSplits,
  };
}

export default function App() {
  const saved = loadSettings();
  const initialModel = MODELS[saved.modelKey] ? saved.modelKey : 'parakeet-tdt-0.6b-v3';
  const [modelKey, setModelKey] = useState(initialModel);
  const [backend, setBackend] = useState(saved.backend || 'webgpu-hybrid');
  const [encoderQuant, setEncoderQuant] = useState(saved.encoderQuant || 'fp16');
  const [decoderQuant, setDecoderQuant] = useState(saved.decoderQuant || 'int8');
  const [preprocessorBackend, setPreprocessorBackend] = useState(saved.preprocessorBackend || 'js');
  const [cpuThreads, setCpuThreads] = useState(Math.max(1, Number(saved.cpuThreads) || 4));
  const [enableProfiling, setEnableProfiling] = useState(saved.enableProfiling !== false);

  const [datasetId, setDatasetId] = useState(saved.datasetId || 'ysdede/parrot-radiology-asr-en');
  const [datasetConfig, setDatasetConfig] = useState(saved.datasetConfig || 'default');
  const [datasetSplit, setDatasetSplit] = useState(saved.datasetSplit || 'train');
  const [sampleCount, setSampleCount] = useState(Math.max(1, Number(saved.sampleCount) || 6));
  const [repeatCount, setRepeatCount] = useState(Math.max(1, Number(saved.repeatCount) || 3));
  const [startOffset, setStartOffset] = useState(Math.max(0, Number(saved.startOffset) || 0));
  const [randomize, setRandomize] = useState(saved.randomize !== false);
  const [randomSeed, setRandomSeed] = useState(saved.randomSeed || '42');

  const [status, setStatus] = useState('Idle');
  const [progressText, setProgressText] = useState('');
  const [progressValue, setProgressValue] = useState(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isPreparing, setIsPreparing] = useState(false);
  const [isRunning, setIsRunning] = useState(false);
  const [modelReady, setModelReady] = useState(false);

  const [configs, setConfigs] = useState(['default']);
  const [configToSplits, setConfigToSplits] = useState({ default: ['train'] });
  const [splitCounts, setSplitCounts] = useState({});
  const [preparedSamples, setPreparedSamples] = useState([]);
  const [runs, setRuns] = useState([]);
  const [lastBatchAt, setLastBatchAt] = useState('');
  const modelRef = useRef(null);

  useEffect(() => {
    saveSettings({
      modelKey,
      backend,
      encoderQuant,
      decoderQuant,
      preprocessorBackend,
      cpuThreads,
      enableProfiling,
      datasetId,
      datasetConfig,
      datasetSplit,
      sampleCount,
      repeatCount,
      startOffset,
      randomize,
      randomSeed,
    });
  }, [
    modelKey,
    backend,
    encoderQuant,
    decoderQuant,
    preprocessorBackend,
    cpuThreads,
    enableProfiling,
    datasetId,
    datasetConfig,
    datasetSplit,
    sampleCount,
    repeatCount,
    startOffset,
    randomize,
    randomSeed,
  ]);

  useEffect(() => () => {
    void modelRef.current?.dispose?.();
  }, []);

  useEffect(() => {
    let active = true;
    setStatus(`Loading dataset metadata for ${datasetId}…`);

    fetchDatasetSplits(datasetId)
      .then((items) => {
        if (!active) return;
        const { configs: nextConfigs, configToSplits: nextConfigToSplits } = collectConfigMap(items);
        const nextConfig = nextConfigs.includes(datasetConfig) ? datasetConfig : nextConfigs[0];
        const nextSplits = nextConfigToSplits[nextConfig] || ['train'];
        const nextSplit = nextSplits.includes(datasetSplit) ? datasetSplit : nextSplits[0];
        const counts = {};

        for (const item of items) {
          if (item.config && item.split && Number.isFinite(item.num_examples)) {
            counts[`${item.config}::${item.split}`] = item.num_examples;
          }
        }

        setConfigs(nextConfigs);
        setConfigToSplits(nextConfigToSplits);
        setSplitCounts(counts);
        if (nextConfig !== datasetConfig) setDatasetConfig(nextConfig);
        if (nextSplit !== datasetSplit) setDatasetSplit(nextSplit);
        setStatus(`Loaded ${items.length} split entries for ${datasetId}`);
      })
      .catch((error) => {
        if (!active) return;
        console.error(error);
        setConfigs(['default']);
        setConfigToSplits({ default: ['train'] });
        setStatus(`Dataset metadata failed: ${error.message}`);
      });

    return () => {
      active = false;
    };
  }, [datasetId]);

  useEffect(() => {
    const availableSplits = configToSplits[datasetConfig] || ['train'];
    if (!availableSplits.includes(datasetSplit)) {
      setDatasetSplit(availableSplits[0]);
    }
  }, [configToSplits, datasetConfig, datasetSplit]);

  const modelConfig = MODELS[modelKey];
  const availableSplits = configToSplits[datasetConfig] || ['train'];
  const summary = useMemo(() => summarizeRuns(runs), [runs]);
  const recentRuns = useMemo(() => runs.slice(-20).reverse(), [runs]);
  const modelLanguages = useMemo(
    () => modelConfig.languages.map((code) => LANGUAGE_NAMES[code] || code).join(', '),
    [modelConfig]
  );

  async function loadModel() {
    setIsLoadingModel(true);
    setModelReady(false);
    setStatus('Downloading model…');
    setProgressText('');
    setProgressValue(0);

    try {
      await modelRef.current?.dispose?.();
      modelRef.current = null;

      const loaded = await loadModelWithFallback({
        repoIdOrModelKey: modelKey,
        options: {
          encoderQuant,
          decoderQuant,
          preprocessor: modelConfig.preprocessor,
          preprocessorBackend,
          backend,
          cpuThreads,
          enableProfiling,
          progress: ({ loaded, total, file }) => {
            const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
            setProgressText(`${file}: ${pct}%`);
            setProgressValue(total > 0 ? pct : 0);
          },
        },
        getParakeetModelFn: getParakeetModel,
        fromUrlsFn: ParakeetModel.fromUrls,
        onBeforeCompile: ({ modelUrls }) => {
          setStatus('Compiling model…');
          setProgressText(`encoder=${modelUrls.quantisation.encoder}, decoder=${modelUrls.quantisation.decoder}`);
          setProgressValue(null);
        },
      });

      modelRef.current = loaded.model;
      setModelReady(true);
      setStatus(loaded.retryUsed ? 'Model ready after FP32 retry' : 'Model ready');
    } catch (error) {
      console.error(error);
      setStatus(`Model load failed: ${error.message}`);
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function prepareDatasetRows() {
    setIsPreparing(true);
    setPreparedSamples([]);
    setStatus('Preparing dataset rows…');

    try {
      const countKey = `${datasetConfig}::${datasetSplit}`;
      const totalRows = Number(splitCounts[countKey]) || 0;
      let rowsPayload = [];

      if (randomize) {
        if (totalRows <= 0) {
          throw new Error('Split size is unknown. Pick a dataset with split metadata or switch off random mode.');
        }
        const randomRows = await fetchRandomRows({
          dataset: datasetId,
          config: datasetConfig,
          split: datasetSplit,
          totalRows,
          sampleCount,
          seed: randomSeed,
        });
        rowsPayload = randomRows.rows;
        setStatus(`Prepared ${randomRows.rows.length} random rows from ${datasetSplit}`);
      } else {
        const sequential = await fetchSequentialRows({
          dataset: datasetId,
          config: datasetConfig,
          split: datasetSplit,
          startOffset,
          limit: sampleCount,
        });
        rowsPayload = sequential.rows;
        setStatus(`Prepared ${sequential.rows.length} sequential rows from ${datasetSplit}`);
      }

      const normalized = rowsPayload
        .map((row, index) => normalizeDatasetRow(row, index))
        .filter((row) => row.audioUrl)
        .map((row, index) => ({
          ...row,
          sampleKey: `${row.rowIndex}:${row.speaker || index + 1}`,
        }));

      setPreparedSamples(normalized);
      if (normalized.length === 0) {
        setStatus('No usable rows were prepared from the selected dataset.');
      }
    } catch (error) {
      console.error(error);
      setStatus(`Dataset preparation failed: ${error.message}`);
    } finally {
      setIsPreparing(false);
    }
  }

  async function runBenchmark() {
    if (!modelRef.current) {
      alert('Load a model first.');
      return;
    }
    if (preparedSamples.length === 0) {
      alert('Prepare dataset rows first.');
      return;
    }

    setIsRunning(true);
    setRuns([]);
    setLastBatchAt(new Date().toISOString());
    setProgressValue(0);
    setProgressText('');

    const nextRuns = [];
    const baselineBySample = new Map();
    const audioCache = new Map();
    const batchId = `batch-${Date.now()}`;
    const totalTasks = preparedSamples.length * repeatCount;
    let finishedTasks = 0;

    try {
      for (let sampleIndex = 0; sampleIndex < preparedSamples.length; sampleIndex += 1) {
        const sample = preparedSamples[sampleIndex];

        if (!audioCache.has(sample.audioUrl)) {
          setStatus(`Decoding ${sample.sampleKey}…`);
          audioCache.set(sample.audioUrl, await decodeAudioSourceToMonoPcm(sample.audioUrl, { targetSampleRate: 16000 }));
        }
        const audio = audioCache.get(sample.audioUrl);

        for (let repeatIndex = 0; repeatIndex < repeatCount; repeatIndex += 1) {
          const startedAt = new Date().toISOString();
          setStatus(`Running ${sample.sampleKey} (${repeatIndex + 1}/${repeatCount})…`);

          try {
            const result = await modelRef.current.transcribe(audio.pcm, audio.sampleRate, {
              returnTimestamps: false,
              returnConfidences: false,
              enableProfiling,
            });
            const firstTranscript = baselineBySample.get(sample.sampleKey) ?? result.utterance_text;
            if (!baselineBySample.has(sample.sampleKey)) {
              baselineBySample.set(sample.sampleKey, firstTranscript);
            }

            nextRuns.push({
              batchId,
              startedAt,
              finishedAt: new Date().toISOString(),
              id: `${sample.sampleKey}-${repeatIndex + 1}`,
              sampleKey: sample.sampleKey,
              sampleOrder: sampleIndex + 1,
              rowIndex: sample.rowIndex,
              repeatIndex: repeatIndex + 1,
              audioDurationSec: audio.durationSec,
              speaker: sample.speaker,
              gender: sample.gender,
              speed: sample.speed,
              volume: sample.volume,
              transcription: result.utterance_text,
              referenceText: sample.referenceText,
              exactMatchToFirst: result.utterance_text === firstTranscript,
              similarityToFirst: textSimilarity(result.utterance_text, firstTranscript),
              metrics: result.metrics ?? null,
              modelKey,
              backend,
              encoderQuant,
              decoderQuant,
              preprocessor: modelConfig.preprocessor,
              preprocessorBackend,
            });
          } catch (error) {
            nextRuns.push({
              batchId,
              startedAt,
              finishedAt: new Date().toISOString(),
              id: `${sample.sampleKey}-${repeatIndex + 1}`,
              sampleKey: sample.sampleKey,
              sampleOrder: sampleIndex + 1,
              rowIndex: sample.rowIndex,
              repeatIndex: repeatIndex + 1,
              audioDurationSec: audio.durationSec,
              speaker: sample.speaker,
              gender: sample.gender,
              speed: sample.speed,
              volume: sample.volume,
              referenceText: sample.referenceText,
              error: error.message,
              modelKey,
              backend,
              encoderQuant,
              decoderQuant,
              preprocessor: modelConfig.preprocessor,
              preprocessorBackend,
            });
          }

          finishedTasks += 1;
          setProgressValue(Math.round((finishedTasks / totalTasks) * 100));
          setProgressText(`${finishedTasks}/${totalTasks} runs complete`);
          setRuns([...nextRuns]);
        }
      }

      setStatus(`Benchmark complete: ${nextRuns.length} runs`);
    } finally {
      setIsRunning(false);
    }
  }

  function exportCsv() {
    if (runs.length === 0) return;
    downloadText(`asrjs-benchmark-${Date.now()}.csv`, benchmarkRunRecordsToCsv(runs), 'text/csv;charset=utf-8');
  }

  function exportJson() {
    if (runs.length === 0) return;
    downloadText(`asrjs-benchmark-${Date.now()}.json`, JSON.stringify({
      settings: {
        modelKey,
        backend,
        encoderQuant,
        decoderQuant,
        preprocessorBackend,
        cpuThreads,
        enableProfiling,
        datasetId,
        datasetConfig,
        datasetSplit,
        sampleCount,
        repeatCount,
        startOffset,
        randomize,
        randomSeed,
      },
      preparedSamples,
      summary,
      runs,
    }, null, 2), 'application/json');
  }

  return (
    <div className="page-shell">
      <div className="page-grid">
        <aside className="panel stack">
          <div className="eyebrow">asr.js benchmark</div>
          <h1>Dataset-driven Parakeet TDT bench</h1>
          <p className="lede">
            Load a Parakeet preset, pull rows from a Hugging Face speech dataset, and benchmark repeatability and timing
            using the shared <code>asr.js</code> helper surface.
          </p>
          <div className="version-pill">asr.js {VERSION} ({SOURCE})</div>

          <section className="card stack">
            <h2>Model</h2>
            <label>
              Model
              <select value={modelKey} onChange={(e) => setModelKey(e.target.value)} disabled={isLoadingModel || modelReady}>
                {Object.entries(MODELS).map(([key, config]) => (
                  <option key={key} value={key}>{config.displayName}</option>
                ))}
              </select>
            </label>
            <p className="hint">{modelConfig.repoId} • {modelLanguages}</p>
            <div className="pair">
              <label>
                Backend
                <select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={isLoadingModel || modelReady}>
                  {BACKENDS.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
              <label>
                CPU threads
                <input type="number" min="1" value={cpuThreads} onChange={(e) => setCpuThreads(Math.max(1, Number(e.target.value) || 1))} disabled={isLoadingModel || modelReady} />
              </label>
            </div>
            <div className="pair">
              <label>
                Encoder
                <select value={encoderQuant} onChange={(e) => setEncoderQuant(e.target.value)} disabled={isLoadingModel || modelReady}>
                  {QUANTS.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
              <label>
                Decoder
                <select value={decoderQuant} onChange={(e) => setDecoderQuant(e.target.value)} disabled={isLoadingModel || modelReady}>
                  {QUANTS.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
            </div>
            <label>
              Preprocessor backend
              <select value={preprocessorBackend} onChange={(e) => setPreprocessorBackend(e.target.value)} disabled={isLoadingModel || modelReady}>
                <option value="js">js</option>
                <option value="onnx">onnx</option>
              </select>
            </label>
            <label className="checkbox-row">
              <input type="checkbox" checked={enableProfiling} onChange={(e) => setEnableProfiling(e.target.checked)} disabled={isLoadingModel || modelReady} />
              <span>Collect stage timing metrics</span>
            </label>
            <button className="primary-button" onClick={loadModel} disabled={isLoadingModel || modelReady}>
              {modelReady ? 'Model ready' : isLoadingModel ? 'Loading…' : 'Load model'}
            </button>
          </section>

          <section className="card stack">
            <h2>Dataset</h2>
            <label>
              Dataset
              <input list="dataset-suggestions" value={datasetId} onChange={(e) => setDatasetId(e.target.value)} disabled={isPreparing || isRunning} />
              <datalist id="dataset-suggestions">
                {DATASET_SUGGESTIONS.map((value) => <option key={value} value={value} />)}
              </datalist>
            </label>
            <div className="pair">
              <label>
                Config
                <select value={datasetConfig} onChange={(e) => setDatasetConfig(e.target.value)} disabled={isPreparing || isRunning}>
                  {configs.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
              <label>
                Split
                <select value={datasetSplit} onChange={(e) => setDatasetSplit(e.target.value)} disabled={isPreparing || isRunning}>
                  {availableSplits.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </label>
            </div>
            <div className="pair">
              <label>
                Samples
                <input type="number" min="1" max="100" value={sampleCount} onChange={(e) => setSampleCount(Math.max(1, Number(e.target.value) || 1))} disabled={isPreparing || isRunning} />
              </label>
              <label>
                Repeats
                <input type="number" min="1" max="20" value={repeatCount} onChange={(e) => setRepeatCount(Math.max(1, Number(e.target.value) || 1))} disabled={isPreparing || isRunning} />
              </label>
            </div>
            <label className="checkbox-row">
              <input type="checkbox" checked={randomize} onChange={(e) => setRandomize(e.target.checked)} disabled={isPreparing || isRunning} />
              <span>Seeded random sample selection</span>
            </label>
            {randomize ? (
              <label>
                Random seed
                <input value={randomSeed} onChange={(e) => setRandomSeed(e.target.value)} disabled={isPreparing || isRunning} />
              </label>
            ) : (
              <label>
                Start offset
                <input type="number" min="0" value={startOffset} onChange={(e) => setStartOffset(Math.max(0, Number(e.target.value) || 0))} disabled={isPreparing || isRunning} />
              </label>
            )}
            <p className="hint">Known rows in split: {Number.isFinite(splitCounts[`${datasetConfig}::${datasetSplit}`]) ? splitCounts[`${datasetConfig}::${datasetSplit}`] : 'unknown'}</p>
            <button className="secondary-button" onClick={prepareDatasetRows} disabled={isPreparing || isRunning}>
              {isPreparing ? 'Preparing…' : 'Prepare benchmark rows'}
            </button>
          </section>

          <section className="card stack">
            <h2>Run</h2>
            <button className="primary-button" onClick={runBenchmark} disabled={!modelReady || preparedSamples.length === 0 || isRunning}>
              {isRunning ? 'Running…' : 'Run benchmark'}
            </button>
            <div className="button-row">
              <button className="secondary-button" onClick={exportCsv} disabled={runs.length === 0}>Export CSV</button>
              <button className="secondary-button" onClick={exportJson} disabled={runs.length === 0}>Export JSON</button>
            </div>
            <div className="status-box">
              <strong>{status}</strong>
              {progressText ? <span>{progressText}</span> : null}
              {progressValue !== null ? (
                <div className="progress-track">
                  <div className="progress-fill" style={{ width: `${progressValue}%` }} />
                </div>
              ) : null}
            </div>
          </section>
        </aside>

        <main className="content stack">
          <section className="hero-card">
            <div>
              <div className="eyebrow">Current batch</div>
              <h2>{lastBatchAt ? new Date(lastBatchAt).toLocaleString() : 'No benchmark run yet'}</h2>
              <p>
                Prepared {preparedSamples.length} samples from <code>{datasetId}</code> / <code>{datasetConfig}</code> /
                <code>{datasetSplit}</code>.
              </p>
            </div>
            <div className="hero-metadata">
              <span>{modelConfig.displayName}</span>
              <span>{backend}</span>
              <span>e:{encoderQuant}</span>
              <span>d:{decoderQuant}</span>
            </div>
          </section>

          <section className="metric-grid">
            <article className="metric-card">
              <div className="metric-label">Runs</div>
              <div className="metric-value">{summary.runCount}</div>
              <div className="metric-sub">{summary.okCount} ok / {summary.errorCount} errors</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Total mean</div>
              <div className="metric-value">{fmtMs(summary.total.mean)}</div>
              <div className="metric-sub">p90 {fmtMs(summary.total.p90)}</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Encode mean</div>
              <div className="metric-value">{fmtMs(summary.encode.mean)}</div>
              <div className="metric-sub">Decode {fmtMs(summary.decode.mean)}</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Repeat exact</div>
              <div className="metric-value">{fmtPct(summary.exactRate)}</div>
              <div className="metric-sub">Similarity {fmtPct(summary.similarity.mean)}</div>
            </article>
            <article className="metric-card">
              <div className="metric-label">Model RTF</div>
              <div className="metric-value">{fmtNumber(summary.rtf.median)}x</div>
              <div className="metric-sub">mean {fmtNumber(summary.rtf.mean)}x</div>
            </article>
          </section>

          <section className="card stack">
            <div className="section-head">
              <h2>Prepared samples</h2>
              <span>{preparedSamples.length}</span>
            </div>
            {preparedSamples.length === 0 ? (
              <p className="empty-text">Prepare rows to preview which dataset samples will be benchmarked.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Sample</th><th>Speaker</th><th>Gender</th><th>Rate</th><th>Reference</th></tr>
                  </thead>
                  <tbody>
                    {preparedSamples.slice(0, 12).map((sample) => (
                      <tr key={sample.sampleKey}>
                        <td>{sample.sampleKey}</td>
                        <td>{sample.speaker || '-'}</td>
                        <td>{sample.gender || '-'}</td>
                        <td>{sample.sampleRate}</td>
                        <td className="text-column">{sample.referenceText || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="card stack">
            <div className="section-head">
              <h2>Recent runs</h2>
              <span>{recentRuns.length}</span>
            </div>
            {recentRuns.length === 0 ? (
              <p className="empty-text">Run a benchmark batch to inspect timings and repeatability.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Run</th><th>Sample</th><th>Repeat</th><th>Total</th><th>Encode</th><th>Decode</th><th>RTF</th><th>Exact</th><th>Sim</th><th>Transcript</th><th>Error</th></tr>
                  </thead>
                  <tbody>
                    {recentRuns.map((run) => (
                      <tr key={run.id}>
                        <td>{run.id}</td>
                        <td>{run.sampleKey}</td>
                        <td>{run.repeatIndex}</td>
                        <td>{fmtMs(run.metrics?.total_ms)}</td>
                        <td>{fmtMs(run.metrics?.encode_ms)}</td>
                        <td>{fmtMs(run.metrics?.decode_ms)}</td>
                        <td>{fmtNumber(run.metrics?.rtf)}x</td>
                        <td>{typeof run.exactMatchToFirst === 'boolean' ? (run.exactMatchToFirst ? 'yes' : 'no') : '-'}</td>
                        <td>{fmtPct(run.similarityToFirst)}</td>
                        <td className="text-column">{run.transcription || '-'}</td>
                        <td className="text-column error-column">{run.error || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section className="card stack">
            <div className="section-head">
              <h2>Sample rollups</h2>
              <span>{preparedSamples.length}</span>
            </div>
            {preparedSamples.length === 0 ? (
              <p className="empty-text">No per-sample rollups yet.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr><th>Sample</th><th>Runs</th><th>Total mean</th><th>Encode mean</th><th>Decode mean</th><th>Similarity mean</th><th>Reference</th></tr>
                  </thead>
                  <tbody>
                    {preparedSamples.map((sample) => {
                      const perSample = runs.filter((run) => run.sampleKey === sample.sampleKey);
                      const ok = perSample.filter((run) => !run.error);
                      const totalMean = mean(ok.map((run) => run.metrics?.total_ms).filter(Number.isFinite));
                      const encodeMean = mean(ok.map((run) => run.metrics?.encode_ms).filter(Number.isFinite));
                      const decodeMean = mean(ok.map((run) => run.metrics?.decode_ms).filter(Number.isFinite));
                      const similarityMean = mean(ok.map((run) => run.similarityToFirst).filter(Number.isFinite));

                      return (
                        <tr key={sample.sampleKey}>
                          <td>{sample.sampleKey}</td>
                          <td>{perSample.length}</td>
                          <td>{fmtMs(totalMean)}</td>
                          <td>{fmtMs(encodeMean)}</td>
                          <td>{fmtMs(decodeMean)}</td>
                          <td>{fmtPct(similarityMean)}</td>
                          <td className="text-column">{sample.referenceText || '-'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
