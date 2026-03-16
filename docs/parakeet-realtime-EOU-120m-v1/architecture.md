Architectural Deep Dive and WebGPU Deployment Analysis: NVIDIA Parakeet-Realtime-EOU-120m-v1
Executive Summary
The migration of highly sophisticated Automatic Speech Recognition (ASR) systems from cloud-tethered infrastructure to edge environments represents one of the most significant engineering challenges in modern machine learning deployment. As developers seek to integrate authentic, real-time conversational agents into web browsers via WebAssembly (WASM) and WebGPU, the architectural demands shift fundamentally from raw throughput to deterministic state management and rigorous memory boundary optimization. The NVIDIA parakeet_realtime_eou_120m-v1 model serves as a paradigm-defining architecture for this exact use case. Explicitly engineered as a streaming speech recognition model with an exceptionally constrained parameter count of 120 million, it is intrinsically designed for voice AI agents operating within a low-latency envelope of 80 to 160 milliseconds.1
A defining operational feature of this model is its native End-of-Utterance (EOU) detection mechanism, which proactively emits a highly specific <EOU> token to signal conversational turn-taking.2 This negates the reliance on external, latency-inducing Voice Activity Detection (VAD) heuristics, allowing upstream agentic logic to execute immediately upon the speaker's natural semantic and prosodic conclusion.4 The underlying architecture relies on a cache-aware FastConformer acoustic encoder paired seamlessly with a Recurrent Neural Network Transducer (RNN-T) decoder.2
Porting this architecture to an ONNX backend for execution within a custom client-side JavaScript library—such as ysdede/parakeet.js—necessitates an exhaustive, surgical deconstruction of its sub-graph components, tensor transformations, and memory layouts. Because WebGPU mandates explicit memory buffer pre-allocation and strict state tracking across compute shaders, the monolithic PyTorch model must be conceptually and practically fractured into isolated execution graphs. This report provides an exhaustive architectural analysis of the preprocessor, pre-encoder, cache-aware conformer, stateful autoregressive decoder, and joint decision network, synthesizing structural parameters, mathematical transformations, and advanced deployment strategies for WebGPU integration.5
The Transition to Cache-Aware Streaming RNN-T Architectures
To fully appreciate the architectural decisions embedded within the Parakeet-Realtime-EOU-120m-v1 model, it is necessary to contextualize the shift away from older acoustic modeling paradigms. Historically, real-time speech recognition relied heavily on Connectionist Temporal Classification (CTC) networks. While computationally efficient, CTC operates under a strict assumption of conditional independence—meaning the probability of emitting a specific token at time step  is entirely independent of the token emitted at time step , given the acoustic features. This limitation forces CTC models to rely heavily on external language models and complex beam-search decoding algorithms to enforce linguistic consistency, which in turn inflates the computational footprint and memory requirements on edge devices.6
The Recurrent Neural Network Transducer (RNN-T) architecture fundamentally resolves the conditional independence limitation by incorporating an explicit autoregressive linguistic model directly into the decoding lattice.2 The RNN-T framework consists of an acoustic encoder processing the audio stream and a prediction network processing the previously emitted non-blank tokens.9 These two representations are then merged through a joint decision network to produce the final vocabulary probability distribution. This autoregressive property ensures that the model natively understands linguistic context, spelling patterns, and syntactic structures without requiring an external language model in the browser.
However, traditional Transformer and Conformer acoustic encoders suffer from quadratic computational complexity with respect to the input sequence length.10 As an audio stream continues indefinitely, the self-attention matrix grows exponentially, quickly exhausting device memory and missing real-time deadlines. The cache-aware streaming configuration directly circumvents this limitation by restricting the attention context to a heavily constrained sliding window.2 Specifically, the Parakeet model enforces a left attention context of 70 frames and a right attention context of 1 frame.2 This creates an infinite-context, finite-memory streaming approach. The model does not retain the entire history of the utterance in its computational graph. Instead, it relies on passing deeply compressed, cached tensors from the previous processing chunk into the current processing chunk, requiring an execution pipeline that is fundamentally loop-based rather than single-shot.11
Audio Ingestion and Preprocessor Dynamics
Before tensors can be dispatched to the neural network within a WebGPU pipeline, raw acoustic data must be ingested, normalized, and transformed into highly specific frequency-domain representations. The preprocessor for the Parakeet-Realtime-EOU-120m-v1 model dictates strict input requirements that must be mirrored flawlessly in the JavaScript and WebAssembly layers.
The initial ingestion layer expects single-channel (mono) audio waveforms captured at a rigid sampling rate of 16,000 Hz (16 kHz).2 Any variation in this sample rate will result in massive spectral misalignment, destroying transcription accuracy. Therefore, the ysdede/parakeet.js library must implement an aggressive resampling algorithm using Web Audio API's OfflineAudioContext or a dedicated WASM resampler before the audio enters the inference pipeline. The minimum valid duration for an inference chunk is exactly 160 milliseconds, which equates to exactly 2560 temporal samples per payload.2 This 160ms window dictates the absolute theoretical floor for the model's latency profile.
The raw audio waveform undergoes a Short-Time Fourier Transform (STFT) to produce a spectrogram. In traditional Python environments, this is handled natively by torchaudio. In a browser environment, relying on the main JavaScript thread to perform complex Fourier transforms introduces severe latency jitter and garbage collection pauses. The optimal integration path requires implementing the STFT logic within a highly optimized WebAssembly module, potentially leveraging SIMD (Single Instruction, Multiple Data) extensions if available in the target browser.
Following the STFT, the power spectrum is mapped onto the Mel scale using a pre-computed filterbank matrix. The exact architectural parameter determined by the model's CoreML metadata conversion scripts is the output dimensionality of the Mel filterbank projection: the mel_dim is explicitly fixed at 128.5 This is significantly higher than older acoustic models that typically utilized 80 Mel bins, indicating that the Parakeet model requires a much finer frequency resolution to capture the acoustic nuances necessary for accurate End-of-Utterance prediction.
The output of the Mel filterbank is then converted to a logarithmic scale to approximate human auditory perception. It is crucial that the WebGPU preprocessor exactly mirrors the normalization statistics used during the model's PyTorch training. Failing to apply per-channel or global layer normalization to the Mel spectrogram prior to feeding the graph will lead to catastrophic degradation in transcription accuracy, often manifesting as hallucinated EOU tokens or catastrophic forgetting in the RNN-T state. The resulting tensor emitted by the preprocessor—and required as the fundamental input to the ONNX graph—has a dimensional shape of $$, where  is the batch size (which is fixed to 1 for client-side edge inference) and  is the number of temporal frames generated by the 160ms chunking logic.5
The Pre-Encoder and 8x Subsampling Mechanism
Once the Mel spectrogram tensor is generated, it enters the acoustic encoder phase. However, the FastConformer architecture is not a monolithic block. It is explicitly bifurcated into a pre-encoding subsampling stage and the primary self-attention blocks. To successfully deploy this model in WebGPU, developers must deeply understand the subsampling mechanics, as they govern the temporal sequence length  moving through the deep attention layers.
To accelerate inference and minimize memory consumption, the FastConformer employs an aggressive downsampling schema.2 Unlike earlier standard conformers that utilized 4x subsampling, the FastConformer utilizes an 8x depthwise convolutional subsampling architecture.7 This subsampling mechanism processes the input spectrogram and outputs tensors with 256 channels before projecting them into the primary 512-dimensional hidden space.5
Because the subsampling involves deep temporal convolutions, it creates a wide receptive field that spans multiple time steps. In a streaming context where data arrives in isolated 160ms chunks, these temporal convolutions cannot operate effectively on isolated fragments. Without access to the previous audio context, the convolutions would experience severe edge artifacts, destroying the temporal continuity of the acoustic features at the boundary of every chunk.
In deployment frameworks like CoreML and ONNX, this continuity is managed by passing a pre_cache tensor explicitly into the encoder wrapper.11 This pre_cache tensor preserves the tail end of the raw Mel frames from the previous 160ms audio chunk. By concatenating the pre_cache with the incoming audio chunk, the convolutions compute accurate features across the boundary, effectively creating a seamless sliding window. The output of this stage is a heavily compressed acoustic embedding where the temporal dimension has been reduced by a factor of 8, preparing the tensor for the deep attention mechanisms.
The FastConformer Acoustic Encoder
Following the subsampler, the sequence passes through the core of the acoustic model: the FastConformer blocks. The structural metadata dictates exactly 17 distinct conformer layers and a primary hidden embedding dimension (hidden_dim) of 512.5
Each of the 17 layers contains feed-forward modules, depthwise convolutions, and multi-head self-attention modules.7 A distinguishing feature of the FastConformer is its reduced convolutional kernel size. While older models relied on massive kernel sizes (often 31) to capture context, the Parakeet model utilizes a significantly reduced convolutional kernel size of 9 within its conformer blocks.7 This reduction drastically lowers the mathematical operations per second (FLOPs) required to execute the layer, directly contributing to the model's viability on lower-power edge devices and integrated browser graphics cards.
The defining feature of the "cache-aware" variant is how the multi-head self-attention calculates its keys and values. As previously established, the model dictates an attention context of ``.2 This implies that for any given time step, the attention mechanism is permitted to "look back" at 70 previous frames and "look ahead" at only 1 future frame. To achieve this across isolated ONNX inference calls without recomputing the entire history of the conversation, the model relies on massive state tensors that must be physically rotated and fed back into the graph by the execution environment.
The metadata identifies these critical architectural requirements as cache_channel_size: 70 and cache_time_size: 8.5 When executing an ONNX session for a single 160ms chunk, the WebGPU execution provider must input the current subsampled embeddings alongside two massive historical tensors: cache_last_channel and cache_last_time.11 These tensors hold the 512-dimensional representations of the previous downsampled frames across all 17 layers. The resulting output embedding represents the final, contextually rich acoustic encoding for the current chunk, ready to be passed to the autoregressive decoder.
FastConformer Structural Parameters
Value / Dimensionality
Description
Number of Layers
17
Depth of the attention and convolution blocks.
Hidden Dimension
512
Size of the acoustic embeddings across all layers.
Attention Left Context
70
Number of historical frames accessible to attention.
Attention Right Context
1
Number of future frames accessible (lookahead).
Convolution Kernel Size
9
Receptive field of the internal depthwise convolutions.

The complexity of porting this to a browser environment lies almost entirely in memory management. Transferring these cache tensors from the GPU back to the CPU (JavaScript context) after every 160ms chunk, only to upload them again for the subsequent chunk, will saturate the PCIe bus and destroy the Real-Time Factor (RTF). The optimal WebGPU implementation involves creating fixed-size ring buffers directly in the GPU's VRAM. Using GPUBuffer.copyBufferToBuffer or custom compute shaders, the library should shift the cache state entirely within VRAM, dramatically lowering the overhead and maintaining the strict 80-160ms latency constraints.
Subword Tokenization and End-of-Utterance Mechanics
The boundary between the continuous acoustic domain and the discrete linguistic domain is managed by the tokenizer. The Parakeet-Realtime-EOU-120m-v1 utilizes a Byte-Pair Encoding (BPE) subword tokenization strategy with a strictly defined vocabulary space. The model is unequivocally English-only and is engineered specifically to output raw text, deliberately omitting both capitalization and punctuation.1 This design choice removes the burden of predicting complex punctuation rules, allowing the 120 million parameters to focus entirely on phonetic accuracy, transcription fidelity, and accurate End-of-Utterance detection.
The architectural metadata unequivocally states that the model has a base vocabulary size (vocab_size) of 1026.5 However, the RNN-T architecture fundamentally requires a "blank" token. The blank token represents time steps where the model has consumed acoustic frames but is not yet confident enough to emit a definitive linguistic output, allowing the temporal alignment to remain fluid. The blank_id is defined exactly as index 1026.5 Therefore, the final linear projection layer of the joint network outputs a probability distribution over 1027 distinct classes (Indices 0 through 1026).5 The associated token vocabulary files (vocab.json and tokenizer.json) map these numerical indices to specific character sequences or subwords.12
The primary distinguishing feature of this specific model iteration is its active participation in dialogue management via the End-of-Utterance token. Emitted under the specific token ID 1024, the <EOU> token serves as an explicit, high-confidence signal to the orchestration layer that the speaker has concluded their thought.3
In traditional streaming setups, voice activity detection (VAD) algorithms monitor audio energy levels, using heuristic silence timers—such as waiting 500 to 1000 milliseconds after the last spoken word—to determine when the user has finished speaking.4 This introduces severe, unavoidable latency into conversational agents. By embedding the EOU detection directly into the RNN-T vocabulary, the model learns the complex prosodic, acoustic, and linguistic cues that indicate the conclusion of a sentence. It outputs the <EOU> token with the exact same latency profile as standard words (80-160ms).2
When the WebGPU decoding loop detects that the argmax of the probabilities is ID 1024, the JavaScript library must trigger several immediate architectural consequences. First, it must emit a turn-taking event to the upstream logic, signaling the LLM or dialogue manager to begin processing the user's completed request. Second, and crucially for memory integrity, it must flush and zero out the FastConformer attention caches (cache_last_channel, cache_last_time).13 If the FastConformer cache retains acoustic resonance from the previous utterance, feeding audio from a new sentence will pollute the attention matrix. Finally, the internal hidden states of the RNN-T prediction network must be reset to zero to clear the linguistic context, ensuring the subsequent turn begins with a clean probabilistic slate.
The Autoregressive Decoder: Stateful RNN-T Mechanics
The decoder module of the Parakeet-Realtime-EOU model entirely abandons linear CTC classification in favor of a complex Recurrent Neural Network Transducer (RNN-T).2 The RNN-T is not a single layer, but rather composed of two distinct sub-networks: the Prediction Network and the Joint Network.9 Porting these to ONNX and WebGPU requires separating them into distinct compute graphs because they operate on entirely different execution loops.5
The Prediction Network
The Prediction Network functions similarly to a highly constrained, causal language model. It accepts the previously emitted non-blank token as input and generates a dense linguistic representation of the textual context.9 The architectural dimensions dictate that the prediction network consists of a single LSTM (Long Short-Term Memory) layer (decoder_layers: 1) with a large hidden dimensionality of 640 (decoder_hidden: 640).5
Because it relies on an LSTM, it possesses an internal hidden state () and cell state (). This statefulness is a primary challenge for streaming ONNX exports.9 During standard operation within the decoding loop, the prediction network is evaluated only when a non-blank token is predicted by the joint network. If a blank token is predicted, the state remains frozen, and the network is not advanced. In the JavaScript orchestration loop, the developer must explicitly allocate and manage these `` tensors using WebGPU buffers or typed arrays. When porting the model to ONNX, the prediction network graph must be explicitly exported with dynamic inputs for the hidden and cell states, ensuring that next_h and next_c are returned as outputs to be preserved in memory and passed back in during the subsequent non-blank token evaluation.
Decoder/Tokenizer Parameters
Value
Operational Role
Vocabulary Size
1026
Total subword tokens available for text generation.
Blank Token ID
1026
Signals the decoder to advance the acoustic time frame.
EOU Token ID
1024
Signals the end of the user's utterance.
Output Classes
1027
Total probabilities generated by the Joint Network.
Prediction Network Type
LSTM
Autoregressive language modeling component.
LSTM Layers
1
Depth of the linguistic prediction network.
LSTM Hidden Dimension
640
Size of the state tracking the conversation history.

The Joint Decision Network
The Joint Network acts as the synthesis engine for the entire architecture. It mathematically merges the acoustic representations emitted by the FastConformer encoder and the linguistic representations emitted by the Prediction Network. It performs a dense projection from the disparate dimensional spaces—512 for the acoustic encoder, 640 for the linguistic decoder—into the final 1027-dimensional vocabulary space.
This merger typically involves adding the projected tensors together, passing them through a non-linear activation function such as a GELU (Gaussian Error Linear Unit) or Tanh, and executing a final linear projection before applying a Softmax function to obtain probabilities.5
The separation of these networks is critically important for WebGPU execution efficiency. The acoustic encoder is executed exactly once per 160ms chunk. However, the Joint Network and the Prediction Network operate in an internal while loop, executing repeatedly for each acoustic frame  generated by the encoder until a blank token is emitted, at which point the loop advances to frame . Separating encoder.onnx and decoder_joint.onnx prevents the catastrophic inefficiency of recomputing the heavy FastConformer layers during the dense token decoding loop.14
Graph Fracturing and ONNX Export Strategy
A naive attempt to export the PyTorch ASRModel instance as a single monolithic ONNX file for streaming inference will unequivocally fail. The internal control flow mechanisms—specifically the dynamic while loops required by the RNN-T decoding lattice—cannot be efficiently represented or updated dynamically inside standard ONNX runtimes or WebGPU shaders without severe performance penalties.9 The standard PyTorch torch.onnx.export utility often throws dynamic axes errors (e.g., RuntimeError: Failed to convert dynamic_axes to dynamic_shapes) when encountering the newer Dynamo engine in PyTorch 2.5+ versions during streaming model exports.9
To successfully prepare the model for ysdede/parakeet.js, the architecture must be surgically split into explicit, static sub-graphs. CoreML conversion methodologies utilized by FluidInference provide an optimal blueprint for this necessary separation.5 The deployment structure must consist of the following discrete ONNX files, each characterized by perfectly static tensor shapes to allow WebGPU to pre-compile optimal compute pipelines.
1. pre_encode.onnx
This graph isolates the convolutional subsampling phase.
Inputs: audio_chunk (Shape: ``), pre_cache (historical audio frames from the previous chunk to support convolutional overlap).
Outputs: subsampled_features, next_pre_cache.
Function: Performs the 8x depthwise convolutional subsampling, reducing the sequence length from 16 frames to 2 frames.
2. conformer_streaming.onnx
This graph houses the heavy multi-head attention blocks. By locking the sequence length to a static chunk size, the tensor shapes become perfectly rigid, preventing pipeline recompilation overheads in the browser.
Inputs: subsampled_features, cache_last_channel (Shape: ), `cache_last_time` (Shape: ).
Outputs: encoded_acoustic_features, next_cache_channel, next_cache_time.
Function: Executes the 17 layers of the FastConformer, utilizing the historical cache to compute the sliding window attention.
3. prediction_network.onnx
This isolates the stateful language model.
Inputs: token_id (Shape: ), `hidden_state_in` (Shape: ), cell_state_in (Shape: ``).
Outputs: linguistic_embedding, hidden_state_out, cell_state_out.
Function: Advances the LSTM based on the last predicted token, yielding a 640-dimensional representation of the sentence structure.
4. joint_network.onnx
This acts as the final fusion block.
Inputs: acoustic_frame (Shape: ), `linguistic_embedding` (Shape: ).
Outputs: logits (Shape: ``).
Function: Merges features, applies non-linearities, and outputs the final vocabulary probabilities.
By separating the architecture into these explicit modules, the JavaScript library assumes total control over the execution flow. The orchestration loop passes the audio through pre_encode and conformer_streaming once per chunk. It then iterates through the resulting acoustic frames, calling joint_network and conditionally calling prediction_network depending entirely on whether the argmax of the logits equates to the blank_id (1026).
WebGPU Memory Management for ysdede/parakeet.js
The primary engineering hurdle when executing parakeet_realtime_eou_120m-v1 in a browser environment via WebAssembly or WebGPU is the severe latency cost associated with moving memory across the CPU-GPU boundary. In a standard CPU-based ONNX execution (such as typical parakeet-rs implementations), the cache tensors are returned to host memory after the conformer_streaming graph executes, and then immediately passed back in for the next chunk.9
For an edge/web implementation, transferring `` 32-bit floats—which amounts to approximately 2.4 Megabytes of data per chunk—back and forth between JavaScript CPU memory and the GPU's VRAM every 160 milliseconds will instantly saturate the memory bus and severely limit the theoretical Real-Time Factor (RTF).
A highly optimized ysdede/parakeet.js implementation must circumvent this by utilizing advanced WebGPU memory architectures. The developer should allocate fixed GPUBuffer objects mapped explicitly for the device, designated to hold cache_last_channel and cache_last_time. The ONNX Web execution provider (or custom WebGPU compute shaders) should be instructed to read from and write directly to these pre-allocated device buffers without ever mapping them back to the CPU memory space.
Through ping-pong buffering, the output of the current iteration becomes the input to the next iteration entirely within the confines of the GPU. The only data payloads that must cross the highly constrained CPU-GPU bus are the initial 160ms audio payload (transformed into the 128-dimensional Mel representation) and the resulting integer token IDs extracted from the joint network. When the JavaScript orchestrator detects the EOU token (1024), it issues a direct WebGPU compute command to rapidly zero-fill the device buffers, instantly resetting the attention mechanism for the next user utterance without expensive data transfers.
Pseudo-Streaming Fallbacks and Latency Trade-offs
If dynamic state management proves too unstable across varying browser implementations, or if WebGPU driver limitations prevent efficient ring-buffer allocations, an alternative deployment topology exists: "Short Batch" pseudo-streaming.17 This methodology was notably explored during the CoreML conversions for Apple Silicon hardware.
In this fallback model, the architecture abandons the complex cache_last_channel mechanics entirely. Instead, the JavaScript library aggregates incoming audio into larger, overlapping rolling windows—typically 1.28 seconds, corresponding exactly to 128 Mel frames.5 The entire 1.28-second chunk is passed through a simplified, stateless version of the FastConformer. Because the chunk is large enough to provide its own internal context, the convolutions and attention mechanisms can operate without cross-chunk caching. The library preserves only the RNN-T LSTM hidden and cell states between chunks to maintain the linguistic context across the rolling windows.17
Streaming Methodology
Latency Profile
Complexity
Architectural Requirement
True Cache-Aware
80 - 160 ms
High
Full GPU buffer rotation, 17-layer state tracking.
Short Batch / Rolling
1.3 seconds
Low
Stateless encoder, large sequence lengths, LSTM state tracking only.

While this pseudo-streaming approach simplifies the encoder execution graph and eliminates the need for massive GPU buffer rotations, it radically inflates processing latency. The latency profile jumps from a theoretical minimum of 80 milliseconds up to 1.3 seconds, directly negating the core competitive advantage of the Realtime EOU architecture. Therefore, while viable as a fallback for unsupported browsers, true cache-aware streaming using isolated sub-graphs remains the architectural gold standard for deploying this model.
Quantitative Performance and Edge Benchmarks
The computational efficiencies achieved through the 8x subsampling and the reduced kernel size translate directly into significant throughput gains, demonstrating the model's viability for edge deployment. When executed on high-end desktop or server hardware, the parakeet-realtime-eou-120m-v1 has demonstrated processing speeds exceeding 429x real-time (RTF).16 While execution within a WebGPU environment will inevitably incur overhead from browser security boundaries and WASM translation, the extremely low baseline parameter count (120M) ensures that it remains performant on consumer-grade graphics architectures.
The model's accuracy profiles are equally robust. Across the standard HuggingFace OpenASR leaderboard datasets, the architecture maintains strong word-level fidelity despite its streaming constraints and missing capitalization.
Dataset Benchmark
Word Error Rate (WER) %
Average Across Sets
6.93%
AMI (Meetings)
11.73%
Earnings22 (Financial)
12.52%
Gigaspeech (Podcasts)
9.66%
Librispeech Test-Clean
2.32%

Note: Benchmark data derived from chunk sizes of ~1.12 seconds.8
The exceptionally low error rate on clean speech (2.32%) confirms that the constrained attention context `` is more than sufficient to resolve complex phonetic ambiguities without requiring full-utterance bidirectional context.
Conclusion
The NVIDIA parakeet_realtime_eou_120m-v1 defines a highly specialized, ruthlessly optimized architectural envelope, perfectly suited for client-side JavaScript execution via WebGPU and ONNX. By abandoning quadratic sequence scaling in favor of a rigid, cache-aware sliding window, and utilizing an 8x downsampling schema with a reduced kernel size, the FastConformer acoustic encoder maintains an exceptionally low memory bandwidth requirement. The decoupling of the temporal alignment through the autoregressive RNN-T decoder ensures that streaming latency is bounded entirely by the 160ms audio ingestion rate, rather than computational bottlenecks.
Furthermore, the integration of the <EOU> token profoundly alters the paradigm of dialogue management. By shifting the responsibility of turn-taking from heuristic algorithms to the acoustic model itself, developers can construct drastically more responsive voice agents.
To successfully implement this architecture within a custom JavaScript library like ysdede/parakeet.js, the monolithic training graph must be rigorously dismantled into isolated static-shape ONNX components. The orchestration logic must shift from simply invoking a model to actively managing states: allocating and rotating the multi-head attention caches within GPU buffers, executing the RNN-T joint network inside an evaluation loop, and decisively zeroing out all acoustic and linguistic memory states the moment the 1024 <EOU> token is observed. When implemented with strict memory-boundary adherence and optimal sub-graph routing, this architecture is uniquely capable of yielding authentic, sub-200ms conversational AI turn-taking directly on consumer edge hardware.
Alıntılanan çalışmalar
Nvidia Parakeet-Realtime-EOU-120m-v1 : r/LocalLLaMA - Reddit, erişim tarihi Mart 16, 2026, https://www.reddit.com/r/LocalLLaMA/comments/1p0okh8/nvidia_parakeetrealtimeeou120mv1/
update for public release · nvidia/parakeet_realtime_eou_120m-v1 at 3aeb2df, erişim tarihi Mart 16, 2026, https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/commit/3aeb2dfa07ab4696531dd374d38625d096e7869f
README.md · nvidia/parakeet_realtime_eou_120m-v1 at 22306ae7e76930747fe5c861716345a02acc83a2 - Hugging Face, erişim tarihi Mart 16, 2026, https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1/blame/22306ae7e76930747fe5c861716345a02acc83a2/README.md
modal-projects/modal-nvidia-asr - GitHub, erişim tarihi Mart 16, 2026, https://github.com/modal-projects/modal-nvidia-asr
metadata.json · FluidInference/parakeet-realtime-eou-120m-coreml ..., erişim tarihi Mart 16, 2026, https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml/blob/main/metadata.json
Models — NVIDIA NeMo Framework User Guide, erişim tarihi Mart 16, 2026, https://docs.nvidia.com/nemo-framework/user-guide/25.02/nemotoolkit/asr/models.html
Models — NVIDIA NeMo Framework User Guide, erişim tarihi Mart 16, 2026, https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
nvidia/nemotron-speech-streaming-en-0.6b - Hugging Face, erişim tarihi Mart 16, 2026, https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
add support for nvidia/parakeet_realtime_eou_120m-v1 #2805 - GitHub, erişim tarihi Mart 16, 2026, https://github.com/k2-fsa/sherpa-onnx/issues/2805
Fast Conformer with Linearly Scalable Attention for Efficient Speech Recognition - arXiv.org, erişim tarihi Mart 16, 2026, https://arxiv.org/html/2305.05084v6
7.61 kB - Hugging Face, erişim tarihi Mart 16, 2026, https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml/resolve/main/convert_streaming_encoder.py?download=true
Upload 35 files · FluidInference/parakeet-realtime-eou-120m-coreml, erişim tarihi Mart 16, 2026, https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml/commit/0b469de48f2da29301c68a429db30ce1de9e82cc
Scaling Real-Time Voice Agents with Cache-Aware Streaming ASR - Hugging Face, erişim tarihi Mart 16, 2026, https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents
parakeet-rs 0.3.3 - Docs.rs, erişim tarihi Mart 16, 2026, https://docs.rs/crate/parakeet-rs/latest/source/README.md
Delete Conversion · FluidInference/parakeet-realtime-eou-120m, erişim tarihi Mart 16, 2026, https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml/commit/dc21317d1d8f6ae3e850fe6f5e3205b4924c81cd
Running Parakeet speech to text on Spark - DGX Spark / GB10 - NVIDIA Developer Forums, erişim tarihi Mart 16, 2026, https://forums.developer.nvidia.com/t/running-parakeet-speech-to-text-on-spark/356353
FluidInference/parakeet-realtime-eou-120m-coreml - Hugging Face, erişim tarihi Mart 16, 2026, https://huggingface.co/FluidInference/parakeet-realtime-eou-120m-coreml
