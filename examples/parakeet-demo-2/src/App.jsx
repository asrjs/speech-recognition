import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  ParakeetModel,
  collectParakeetLocalEntries,
  createParakeetLocalEntries,
  decodeAudioSourceToMonoPcm,
  getParakeetModel,
  inspectParakeetLocalEntries,
  loadModelWithFallback,
  loadParakeetModelFromLocalEntries,
  MODELS,
  LANGUAGE_NAMES,
} from 'asr.js';
import './App.css';

const SETTINGS_STORAGE_KEY = 'asrjs.parakeet-inspector.settings.v1';
const MODEL_SOURCE_OPTIONS = { HUGGINGFACE: 'huggingface', LOCAL: 'local' };
const QUANT_OPTIONS = ['fp16', 'int8', 'fp32'];
const SAMPLE_URL = `${import.meta.env.BASE_URL || '/'}assets/Harvard-L2-1.ogg`;
const VERSION = typeof __ASRJS_VERSION__ !== 'undefined' ? __ASRJS_VERSION__ : 'unknown';
const SOURCE = typeof __ASRJS_SOURCE__ !== 'undefined' ? __ASRJS_SOURCE__ : 'dev';

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_STORAGE_KEY) || '{}');
  } catch {
    return {};
  }
}

function saveSettings(settings) {
  try {
    localStorage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify(settings));
  } catch {
    // Ignore storage failures.
  }
}

function toJSON(value) {
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function average(values) {
  const valid = values.filter(Number.isFinite);
  if (valid.length === 0) return null;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function fmtMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)} ms` : '-';
}

function fmtRtfx(value) {
  return Number.isFinite(value) ? `${value.toFixed(2)}x` : '-';
}

function summarizeBenchmark(runs, durationSec) {
  const avgWallMs = average(runs.map((run) => run.wallMs));
  return {
    count: runs.length,
    durationSec,
    avgWallMs,
    avgWallRtfx: Number.isFinite(avgWallMs) && durationSec > 0 ? durationSec / (avgWallMs / 1000) : null,
    avgEncodeMs: average(runs.map((run) => run.metrics?.encode_ms ?? NaN)),
    avgDecodeMs: average(runs.map((run) => run.metrics?.decode_ms ?? NaN)),
    avgTotalMs: average(runs.map((run) => run.metrics?.total_ms ?? NaN)),
  };
}

export default function App() {
  const initialSettings = loadSettings();
  const initialModel = MODELS[initialSettings.selectedModel] ? initialSettings.selectedModel : 'parakeet-tdt-0.6b-v2';
  const [modelSource, setModelSource] = useState(initialSettings.modelSource || MODEL_SOURCE_OPTIONS.HUGGINGFACE);
  const [selectedModel, setSelectedModel] = useState(initialModel);
  const [backend, setBackend] = useState(initialSettings.backend || 'webgpu-hybrid');
  const [encoderQuant, setEncoderQuant] = useState(initialSettings.encoderQuant || 'fp16');
  const [decoderQuant, setDecoderQuant] = useState(initialSettings.decoderQuant || 'int8');
  const [preprocessor, setPreprocessor] = useState(initialSettings.preprocessor || 'nemo128');
  const [preprocessorBackend, setPreprocessorBackend] = useState(initialSettings.preprocessorBackend || 'js');
  const [cpuThreads, setCpuThreads] = useState(Math.max(1, Number(initialSettings.cpuThreads) || 4));
  const [benchmarkRepeats, setBenchmarkRepeats] = useState(Math.max(1, Number(initialSettings.benchmarkRepeats) || 3));
  const [returnWords, setReturnWords] = useState(initialSettings.returnWords !== false);
  const [returnTokenIds, setReturnTokenIds] = useState(Boolean(initialSettings.returnTokenIds));
  const [returnFrameIndices, setReturnFrameIndices] = useState(Boolean(initialSettings.returnFrameIndices));
  const [returnLogProbs, setReturnLogProbs] = useState(Boolean(initialSettings.returnLogProbs));
  const [returnTdtSteps, setReturnTdtSteps] = useState(Boolean(initialSettings.returnTdtSteps));
  const [enableProfiling, setEnableProfiling] = useState(initialSettings.enableProfiling !== false);
  const [status, setStatus] = useState('Idle');
  const [progressText, setProgressText] = useState('');
  const [progressPct, setProgressPct] = useState(null);
  const [isLoadingModel, setIsLoadingModel] = useState(false);
  const [isWorking, setIsWorking] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [resultJson, setResultJson] = useState('');
  const [latestMetrics, setLatestMetrics] = useState(null);
  const [benchmarkRuns, setBenchmarkRuns] = useState([]);
  const [benchmarkSummary, setBenchmarkSummary] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioLabel, setAudioLabel] = useState('Bundled sample');
  const [audioUrl, setAudioUrl] = useState(SAMPLE_URL);
  const [localEntries, setLocalEntries] = useState([]);
  const [localFolderName, setLocalFolderName] = useState('');
  const [localTokenizerOptions, setLocalTokenizerOptions] = useState(['vocab.txt']);
  const [localTokenizerName, setLocalTokenizerName] = useState('vocab.txt');
  const [localPreprocessorOptions, setLocalPreprocessorOptions] = useState(['nemo128']);
  const [encoderOptions, setEncoderOptions] = useState(QUANT_OPTIONS);
  const [decoderOptions, setDecoderOptions] = useState(QUANT_OPTIONS);
  const modelRef = useRef(null);
  const audioInputRef = useRef(null);
  const folderInputRef = useRef(null);

  useEffect(() => {
    saveSettings({
      modelSource,
      selectedModel,
      backend,
      encoderQuant,
      decoderQuant,
      preprocessor,
      preprocessorBackend,
      cpuThreads,
      benchmarkRepeats,
      returnWords,
      returnTokenIds,
      returnFrameIndices,
      returnLogProbs,
      returnTdtSteps,
      enableProfiling,
    });
  }, [modelSource, selectedModel, backend, encoderQuant, decoderQuant, preprocessor, preprocessorBackend, cpuThreads, benchmarkRepeats, returnWords, returnTokenIds, returnFrameIndices, returnLogProbs, returnTdtSteps, enableProfiling]);

  useEffect(() => () => {
    if (audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl);
  }, [audioUrl]);

  useEffect(() => () => {
    void modelRef.current?.dispose?.();
  }, []);

  const modelOptions = useMemo(
    () => Object.entries(MODELS).map(([key, config]) => ({ key, config })),
    []
  );

  async function handleLocalEntries(entries, folderName) {
    const inspected = inspectParakeetLocalEntries(entries);
    const nextEncoderOptions = inspected.encoderQuantizations.length > 0 ? [...inspected.encoderQuantizations] : ['fp32'];
    const nextDecoderOptions = inspected.decoderQuantizations.length > 0 ? [...inspected.decoderQuantizations] : ['fp32'];
    const nextTokenizerOptions = inspected.tokenizerNames.length > 0 ? [...inspected.tokenizerNames] : ['vocab.txt'];
    const nextPreprocessorOptions = inspected.preprocessorNames.length > 0 ? [...inspected.preprocessorNames] : ['nemo128'];
    setLocalEntries(entries);
    setLocalFolderName(folderName || '');
    setEncoderOptions(nextEncoderOptions);
    setDecoderOptions(nextDecoderOptions);
    setLocalTokenizerOptions(nextTokenizerOptions);
    setLocalPreprocessorOptions(nextPreprocessorOptions);
    setEncoderQuant(nextEncoderOptions[0]);
    setDecoderQuant(nextDecoderOptions[0]);
    setLocalTokenizerName(nextTokenizerOptions[0]);
    setPreprocessor(nextPreprocessorOptions[0]);
    setStatus(`Selected local model folder (${entries.length} files)`);
  }

  async function pickLocalFolder() {
    try {
      if (typeof window.showDirectoryPicker === 'function') {
        const dirHandle = await window.showDirectoryPicker({ mode: 'read' });
        await handleLocalEntries(await collectParakeetLocalEntries(dirHandle), dirHandle.name);
      } else {
        folderInputRef.current?.click();
      }
    } catch (error) {
      if (error?.name !== 'AbortError') setStatus(`Folder read failed: ${error.message}`);
    }
  }

  async function loadModel() {
    setIsLoadingModel(true);
    setModelLoaded(false);
    setStatus(modelSource === MODEL_SOURCE_OPTIONS.LOCAL ? 'Preparing local model…' : 'Downloading model…');
    setProgressText('');
    setProgressPct(modelSource === MODEL_SOURCE_OPTIONS.LOCAL ? null : 0);

    try {
      await modelRef.current?.dispose?.();
      modelRef.current = null;

      if (modelSource === MODEL_SOURCE_OPTIONS.LOCAL) {
        if (localEntries.length === 0) throw new Error('Pick a local model folder first.');
        const localResult = await loadParakeetModelFromLocalEntries(localEntries, {
          modelId: selectedModel,
          encoderQuant,
          decoderQuant,
          tokenizerName: localTokenizerName,
          preprocessorName: preprocessor,
          preprocessorBackend,
          backend,
          cpuThreads,
          enableProfiling,
        });
        modelRef.current = localResult.model;
        setProgressText(`${localResult.selection.encoderName} + ${localResult.selection.decoderName}`);
      } else {
        const remoteResult = await loadModelWithFallback({
          repoIdOrModelKey: selectedModel,
          options: {
            encoderQuant,
            decoderQuant,
            preprocessor,
            preprocessorBackend,
            backend,
            cpuThreads,
            enableProfiling,
            progress: ({ loaded, total, file }) => {
              setProgressText(`${file}: ${Math.round(total > 0 ? (loaded / total) * 100 : 0)}%`);
              setProgressPct(total > 0 ? Math.round((loaded / total) * 100) : 0);
            },
          },
          getParakeetModelFn: getParakeetModel,
          fromUrlsFn: ParakeetModel.fromUrls,
          onBeforeCompile: ({ modelUrls }) => {
            setStatus('Compiling model…');
            setProgressText(`encoder=${modelUrls.quantisation.encoder}, decoder=${modelUrls.quantisation.decoder}`);
            setProgressPct(null);
          },
        });
        modelRef.current = remoteResult.model;
      }

      setModelLoaded(true);
      setStatus('Model ready');
    } catch (error) {
      console.error(error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setIsLoadingModel(false);
    }
  }

  async function transcribeCurrentAudio() {
    if (!modelRef.current) return alert('Load a model first.');
    setIsWorking(true);
    setStatus('Preparing audio…');
    try {
      const prepared = await decodeAudioSourceToMonoPcm(audioFile || SAMPLE_URL, { targetSampleRate: 16000 });
      setStatus('Transcribing…');
      const result = await modelRef.current.transcribe(prepared.pcm, prepared.sampleRate, {
        returnTimestamps: returnWords,
        returnConfidences: true,
        returnTokenIds,
        returnFrameIndices,
        returnLogProbs,
        returnTdtSteps,
        enableProfiling,
      });
      setTranscript(result.utterance_text);
      setResultJson(toJSON(result));
      setLatestMetrics({ ...result.metrics, durationSec: prepared.durationSec, words: result.words?.length ?? 0, tokens: result.tokens?.length ?? 0 });
      setStatus('Transcription complete');
    } catch (error) {
      console.error(error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setIsWorking(false);
    }
  }

  async function benchmarkCurrentAudio() {
    if (!modelRef.current) return alert('Load a model first.');
    setIsWorking(true);
    setStatus('Preparing benchmark audio…');
    try {
      const prepared = await decodeAudioSourceToMonoPcm(audioFile || SAMPLE_URL, { targetSampleRate: 16000 });
      const runs = [];
      for (let index = 0; index < benchmarkRepeats; index += 1) {
        setStatus(`Benchmark ${index + 1}/${benchmarkRepeats}…`);
        const start = performance.now();
        const result = await modelRef.current.transcribe(prepared.pcm, prepared.sampleRate, {
          returnTimestamps: returnWords,
          returnConfidences: true,
          returnTokenIds,
          returnFrameIndices,
          returnLogProbs,
          returnTdtSteps,
          enableProfiling,
        });
        runs.push({
          index: index + 1,
          wallMs: performance.now() - start,
          metrics: result.metrics ?? null,
          words: result.words?.length ?? 0,
          tokens: result.tokens?.length ?? 0,
        });
      }
      setBenchmarkRuns(runs);
      setBenchmarkSummary(summarizeBenchmark(runs, prepared.durationSec));
      setStatus('Benchmark complete');
    } catch (error) {
      console.error(error);
      setStatus(`Failed: ${error.message}`);
    } finally {
      setIsWorking(false);
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100">
      <div className="max-w-7xl mx-auto px-4 py-8 lg:px-8">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4 mb-8">
          <div>
            <p className="text-xs uppercase tracking-[0.35em] text-gray-500 dark:text-gray-400 mb-2">asr.js demo</p>
            <h1 className="text-3xl font-semibold tracking-tight">Parakeet Inspector</h1>
            <p className="mt-2 text-sm text-gray-600 dark:text-gray-400 max-w-3xl">
              A clean <code className="font-mono">asr.js</code> demo for Parakeet TDT loading, direct native transcripts,
              local-folder artifacts, and repeatable browser inference checks.
            </p>
          </div>
          <div className="text-xs font-mono px-3 py-2 rounded-full border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900">
            asr.js {VERSION} ({SOURCE})
          </div>
        </div>

        <div className="grid gap-6 xl:grid-cols-[360px,minmax(0,1fr)]">
          <aside className="space-y-6">
            <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5 shadow-sm space-y-4">
              <h2 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400">Model</h2>
              <select value={modelSource} onChange={(e) => setModelSource(e.target.value)} disabled={isLoadingModel || modelLoaded} className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                <option value={MODEL_SOURCE_OPTIONS.HUGGINGFACE}>Hugging Face</option>
                <option value={MODEL_SOURCE_OPTIONS.LOCAL}>Local folder</option>
              </select>
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)} disabled={isLoadingModel || modelLoaded} className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                {modelOptions.map(({ key, config }) => <option key={key} value={key}>{config.displayName}</option>)}
              </select>
              <p className="text-xs text-gray-500 dark:text-gray-400">{MODELS[selectedModel].repoId} • {MODELS[selectedModel].languages.map((lang) => LANGUAGE_NAMES[lang] || lang).join(', ')}</p>
              {modelSource === MODEL_SOURCE_OPTIONS.LOCAL && (
                <>
                  <button onClick={pickLocalFolder} disabled={isLoadingModel || modelLoaded} className="w-full rounded-lg bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 px-4 py-2 text-sm font-medium">Pick local model folder</button>
                  <input ref={folderInputRef} type="file" multiple webkitdirectory="" directory="" onChange={(e) => {
                    const files = Array.from(e.target.files || []);
                    if (files.length === 0) return;
                    void handleLocalEntries(createParakeetLocalEntries(files), '');
                    e.target.value = '';
                  }} className="hidden" />
                  <p className="text-xs text-gray-500 dark:text-gray-400">{localFolderName || 'No folder selected yet.'}</p>
                </>
              )}
              <div className="grid grid-cols-2 gap-3">
                <select value={backend} onChange={(e) => setBackend(e.target.value)} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  <option value="webgpu-hybrid">webgpu-hybrid</option>
                  <option value="webgpu">webgpu</option>
                  <option value="webgpu-strict">webgpu-strict</option>
                  <option value="wasm">wasm</option>
                </select>
                <input type="number" min="1" value={cpuThreads} onChange={(e) => setCpuThreads(Math.max(1, Number(e.target.value) || 1))} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <select value={encoderQuant} onChange={(e) => setEncoderQuant(e.target.value)} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  {encoderOptions.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
                <select value={decoderQuant} onChange={(e) => setDecoderQuant(e.target.value)} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  {decoderOptions.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <select value={preprocessorBackend} onChange={(e) => setPreprocessorBackend(e.target.value)} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  <option value="js">js</option>
                  <option value="onnx">onnx</option>
                </select>
                <select value={preprocessor} onChange={(e) => setPreprocessor(e.target.value)} disabled={isLoadingModel || modelLoaded} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  {(modelSource === MODEL_SOURCE_OPTIONS.LOCAL ? localPreprocessorOptions : ['nemo128', 'nemo80']).map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              </div>
              {modelSource === MODEL_SOURCE_OPTIONS.LOCAL && (
                <select value={localTokenizerName} onChange={(e) => setLocalTokenizerName(e.target.value)} disabled={isLoadingModel || modelLoaded} className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm">
                  {localTokenizerOptions.map((value) => <option key={value} value={value}>{value}</option>)}
                </select>
              )}
              <div className="grid grid-cols-2 gap-2 text-sm">
                {[
                  ['Words', returnWords, setReturnWords],
                  ['Token ids', returnTokenIds, setReturnTokenIds],
                  ['Frame idx', returnFrameIndices, setReturnFrameIndices],
                  ['Log probs', returnLogProbs, setReturnLogProbs],
                  ['TDT steps', returnTdtSteps, setReturnTdtSteps],
                  ['Profiling', enableProfiling, setEnableProfiling],
                ].map(([label, checked, setter]) => (
                  <label key={label} className="flex items-center justify-between gap-2 rounded-lg border border-gray-200 dark:border-gray-800 px-3 py-2">
                    <span>{label}</span>
                    <input type="checkbox" checked={checked} onChange={(e) => setter(e.target.checked)} className="toggle-checkbox w-4 h-4 rounded border-gray-300 dark:border-gray-700" />
                  </label>
                ))}
              </div>
              <button onClick={loadModel} disabled={isLoadingModel || modelLoaded} className="w-full rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white px-4 py-3 font-medium disabled:opacity-50">{modelLoaded ? 'Model loaded' : isLoadingModel ? 'Loading…' : 'Load model'}</button>
            </section>

            <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5 shadow-sm space-y-4">
              <h2 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400">Audio</h2>
              <div className="flex gap-2">
                <button onClick={() => { if (audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl); setAudioFile(null); setAudioLabel('Bundled sample'); setAudioUrl(SAMPLE_URL); }} className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm">Bundled sample</button>
                <button onClick={() => audioInputRef.current?.click()} className="flex-1 rounded-lg border border-gray-300 dark:border-gray-700 px-3 py-2 text-sm">Upload</button>
                <input ref={audioInputRef} type="file" accept="audio/*" onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (!file) return;
                  if (audioUrl.startsWith('blob:')) URL.revokeObjectURL(audioUrl);
                  setAudioFile(file);
                  setAudioLabel(file.name);
                  setAudioUrl(URL.createObjectURL(file));
                  e.target.value = '';
                }} className="hidden" />
              </div>
              <div className="rounded-xl border border-dashed border-gray-300 dark:border-gray-700 p-3 text-sm text-gray-500 dark:text-gray-400">{audioLabel}</div>
              <audio controls src={audioUrl} className="w-full" />
              <input type="number" min="1" max="50" value={benchmarkRepeats} onChange={(e) => setBenchmarkRepeats(Math.max(1, Number(e.target.value) || 1))} className="w-full rounded-lg border border-gray-300 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 px-3 py-2 text-sm" />
              <div className="grid grid-cols-2 gap-3">
                <button onClick={transcribeCurrentAudio} disabled={!modelLoaded || isWorking} className="rounded-xl bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 px-4 py-3 font-medium disabled:opacity-50">{isWorking ? 'Working…' : 'Transcribe'}</button>
                <button onClick={benchmarkCurrentAudio} disabled={!modelLoaded || isWorking} className="rounded-xl border border-gray-300 dark:border-gray-700 px-4 py-3 font-medium disabled:opacity-50">Benchmark</button>
              </div>
            </section>

            <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5 shadow-sm">
              <h2 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400 mb-3">Status</h2>
              <p className={`text-sm font-medium ${modelLoaded ? 'text-emerald-600 dark:text-emerald-400' : 'text-gray-600 dark:text-gray-400'}`}>{status}</p>
              {progressText && <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">{progressText}</p>}
              {progressPct !== null && <div className="mt-3 h-1.5 rounded-full bg-gray-200 dark:bg-gray-800 overflow-hidden"><div className="h-full bg-emerald-500" style={{ width: `${progressPct}%` }} /></div>}
            </section>
          </aside>

          <main className="space-y-6">
            <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 shadow-sm">
              <h2 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400 mb-4">Transcript</h2>
              <div className="rounded-xl bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 p-5 min-h-[160px]">
                <p className="text-lg leading-relaxed">{transcript || 'Run a transcription to inspect the current direct Parakeet output.'}</p>
              </div>
            </section>

            <section className="grid gap-6 lg:grid-cols-3">
              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5 shadow-sm">
                <h3 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400 mb-4">Latest metrics</h3>
                {latestMetrics ? (
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between"><span>Duration</span><span>{latestMetrics.durationSec?.toFixed(2)} s</span></div>
                    <div className="flex justify-between"><span>Total</span><span>{fmtMs(latestMetrics.total_ms)}</span></div>
                    <div className="flex justify-between"><span>Encode</span><span>{fmtMs(latestMetrics.encode_ms)}</span></div>
                    <div className="flex justify-between"><span>Decode</span><span>{fmtMs(latestMetrics.decode_ms)}</span></div>
                    <div className="flex justify-between"><span>RTF</span><span>{fmtRtfx(latestMetrics.rtf)}</span></div>
                    <div className="flex justify-between"><span>Words</span><span>{latestMetrics.words}</span></div>
                    <div className="flex justify-between"><span>Tokens</span><span>{latestMetrics.tokens}</span></div>
                  </div>
                ) : <p className="text-sm text-gray-500 dark:text-gray-400">No transcription yet.</p>}
              </div>

              <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-5 shadow-sm lg:col-span-2">
                <h3 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400 mb-4">Benchmark</h3>
                {benchmarkSummary ? (
                  <div className="space-y-4">
                    <div className="grid sm:grid-cols-4 gap-3">
                      {[
                        ['Runs', benchmarkSummary.count],
                        ['Avg wall', fmtMs(benchmarkSummary.avgWallMs)],
                        ['Avg wall RTFx', fmtRtfx(benchmarkSummary.avgWallRtfx)],
                        ['Avg model total', fmtMs(benchmarkSummary.avgTotalMs)],
                      ].map(([label, value]) => (
                        <div key={label} className="rounded-xl bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 p-3">
                          <p className="text-[11px] uppercase tracking-[0.2em] text-gray-500 dark:text-gray-400">{label}</p>
                          <p className="mt-2 text-sm font-medium">{value}</p>
                        </div>
                      ))}
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead className="text-left text-gray-500 dark:text-gray-400">
                          <tr><th className="pb-2 pr-4">Run</th><th className="pb-2 pr-4">Wall</th><th className="pb-2 pr-4">Encode</th><th className="pb-2 pr-4">Decode</th><th className="pb-2">Tokens</th></tr>
                        </thead>
                        <tbody>
                          {benchmarkRuns.map((run) => (
                            <tr key={run.index} className="border-t border-gray-200 dark:border-gray-800">
                              <td className="py-2 pr-4">#{run.index}</td>
                              <td className="py-2 pr-4">{fmtMs(run.wallMs)}</td>
                              <td className="py-2 pr-4">{fmtMs(run.metrics?.encode_ms)}</td>
                              <td className="py-2 pr-4">{fmtMs(run.metrics?.decode_ms)}</td>
                              <td className="py-2">{run.tokens}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : <p className="text-sm text-gray-500 dark:text-gray-400">Run repeated inference to inspect stability and throughput.</p>}
              </div>
            </section>

            <section className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 p-6 shadow-sm">
              <h2 className="text-xs font-bold uppercase tracking-[0.25em] text-gray-500 dark:text-gray-400 mb-4">Native output</h2>
              <pre className="overflow-x-auto rounded-xl bg-gray-50 dark:bg-gray-950 border border-gray-200 dark:border-gray-800 p-4 text-xs leading-6 text-gray-700 dark:text-gray-300">{resultJson || '{\n  "status": "run a transcription"\n}'}</pre>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}
