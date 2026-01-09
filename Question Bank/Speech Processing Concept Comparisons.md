
Based on standard "Question Bank" topics for Speech Recognition (DSAI-456), here are comparisons of key concepts derived from Lectures 1-11.

## 1. Acoustic Modeling: HMM vs. DNN

This is a classic comparison regarding how we model the probability of speech units (phonemes) given acoustic features.

|Feature|Hidden Markov Model (HMM) - GMM|Deep Neural Network (DNN)|
|---|---|---|
|**Core Principle**|Uses **Gaussian Mixture Models (GMMs)** to model the probability distribution of acoustic features for each state.|Uses **multiple hidden layers** of neurons to learn complex, non-linear transformations of input features.|
|**Input Features**|Requires **decorrelated** features (like **MFCCs**) because GMMs typically assume diagonal covariance matrices to remain computationally tractable.|Can handle **correlated** features (like **Filterbanks/Spectrograms**) directly; the layers learn to decorrelate and extract features automatically.|
|**Context Window**|Typically looks at a **single frame** (or very narrow window) of speech at a time, relying on the Markov assumption (state depends only on previous state).|Can ingest a **wide context window** (e.g., 10-20 frames left and right) to capture temporal dynamics and co-articulation effects directly at the input level.|
|**Data Efficiency**|Performs decently with **smaller datasets**; converges faster but plateaus in accuracy. Good for low-resource languages.|Requires **large amounts of training data** to generalize well but achieves significantly higher accuracy (state-of-the-art before End-to-End models).|
|**Generative vs. Discriminative**|**Generative**: Models the joint probability $P(X, Y)$ (how data is generated). Can theoretically generate synthetic speech features.|**Discriminative**: Models the posterior probability $P(Y \mid X)$ (classifying the state given data). Focuses purely on the decision boundary between classes.|
|**Invariance**|**Low**: Sensitive to speaker variations and environmental noise unless explicitly adapted (MLLR/MAP).|**High**: The hidden layers learn invariant representations, making it more robust to speaker accents and noise.|

## 2. Feature Extraction: MFCC vs. Mel-Spectrogram

A comparison of the two most common ways to represent audio signals before feeding them into a model.

|Feature|MFCC (Mel-Frequency Cepstral Coefficients)|Mel-Spectrogram (Filterbanks)|
|---|---|---|
|**Computation**|Mel-Spectrogram $\rightarrow$ Log $\rightarrow$ **Discrete Cosine Transform (DCT)**.|FFT $\rightarrow$ Mel-Scale Filtering $\rightarrow$ Logarithm.|
|**Correlation**|**Decorrelated**: The DCT step removes correlation between dimensions, making it ideal for GMMs (diagonal covariance).|**Correlated**: Adjacent frequency bins are highly correlated, preserving the spectral structure.|
|**Best Used With**|**GMM-HMMs** and legacy systems. Also used when data size is very small.|**CNNs / DNNs / Transformers** (Neural networks effectively learn correlations and local patterns).|
|**Information Loss**|**Higher**: DCT discards some information (higher "quefrencies") to compress the signal.|**Lower**: Retains more physical detail of the sound.|
|**Visualization**|Hard for humans to interpret visually.|Easy to interpret (looks like a "heatmap" of sound over time).|

## 3. Metric: WER vs. CER

Common metrics for evaluating Speech Recognition systems.

|Metric|WER (Word Error Rate)|CER (Character Error Rate)|
|---|---|---|
|**Unit of Error**|Whole **Words**.|Individual **Characters** (letters).|
|**Formula**|$(S + D + I) / N_{words}$|$(S + D + I) / N_{chars}$|
|**Difficulty**|Generally **higher** (harder to get right). A single wrong letter makes the whole word wrong.|Generally **lower**.|
|**Use Case**|Standard for general speech recognition (Siri, Google Assistant).|Used for languages with complex scripts (e.g., Chinese, Japanese) or poor spelling models.|

## 4. Signal Analysis: Narrowband vs. Wideband Spectrograms

This comparison focuses on the trade-off between time and frequency resolution (Heisenberg uncertainty principle in signal processing).

|Feature|Narrowband Spectrogram|Wideband Spectrogram|
|---|---|---|
|**Window Length**|**Long** window (e.g., 20ms - 30ms or more).|**Short** window (e.g., 3ms - 5ms).|
|**Frequency Resolution**|**High**: Can distinguish individual harmonics.|**Low**: Smears harmonics together.|
|**Time Resolution**|**Low**: Smears rapid changes in time.|**High**: Can detect precise timing of events (like bursts).|
|**Visual Structure**|Shows **horizontal striations** (harmonics of the pitch).|Shows **vertical striations** (individual glottal pulses/pitch periods) and **Formants** clearly.|
|**Primary Use**|Pitch (F0) tracking, analyzing harmonic structure.|Formant analysis, boundary detection, phone segmentation.|

## 5. End-to-End ASR: CTC vs. Attention (LAS)

Comparison of architectures that replaced the traditional HMM-DNN hybrid approach.

|Feature|CTC (Connectionist Temporal Classification)|Attention-Based (e.g., LAS, Transformer)|
|---|---|---|
|**Alignment**|**Hard/Implicit**: Uses a "blank" token and dynamic programming to handle alignment.|**Soft/Explicit**: Uses an attention mechanism to weigh input frames relevant to the current output.|
|**Independence**|**Conditional Independence**: Assumes outputs are independent given the input (bad for language modeling).|**Auto-regressive**: Prediction at step $t$ depends on predictions at $t-1$ (learns implicit Language Model).|
|**Monotonicity**|Strictly **monotonic** (speech naturally moves forward in time).|Can be **non-monotonic** (potentially problematic for speech, requires constraints).|
|**Streaming**|Easier to adapt for **streaming/online** recognition.|Harder to stream (often requires seeing the whole utterance).|
|**Pros/Cons**|Fast decoding, but often needs an external Language Model.|Higher accuracy, but computationally heavier and slower decoding.|

## 6. Speaker Tasks: Verification vs. Identification

Distinguishing between the two main biometric tasks in speech processing.

|Feature|Speaker Verification (SV)|Speaker Identification (SI)|
|---|---|---|
|**The Question**|"Is this person who they **claim** to be?"|"Who **is** this person?"|
|**Mapping**|**1:1** (One-to-One) comparison.|**1:N** (One-to-Many) classification.|
|**Input**|Test speech + Claimed Identity ID.|Test speech only.|
|**Output**|Binary (Accept / Reject) based on a threshold.|A specific User ID (or "Unknown").|
|**Difficulty**|Constant complexity (does not grow with user base size).|Difficulty increases as the number of users ($N$) grows.|
|**Example**|"Voice Match" to unlock your phone.|Police forensics (identifying a suspect from a database).|

## 7. Feature Analysis: LPC vs. MFCC

A comparison of older vs. modern feature extraction techniques.

|Feature|LPC (Linear Predictive Coding)|MFCC (Mel-Frequency Cepstral Coefficients)|
|---|---|---|
|**Model Basis**|**Production**: Models the human vocal tract as a tube (source-filter model).|**Perception**: Models the human ear/auditory system (Mel scale).|
|**Robustness**|**Low**: Very sensitive to background noise and quantization errors.|**High**: Cepstral mean subtraction helps remove channel noise.|
|**Mathematical Goal**|Minimizes the mean squared error between predicted and actual speech samples.|Decorrelates the spectral energy to compress information.|
|**Primary Use**|**Speech Coding/Compression** (GSM, VoIP) & Synthesis.|**Speech Recognition** (ASR) & Speaker Identification.|

## 8. TTS Paradigm: VALL-E vs. Traditional (Tacotron/FastSpeech)

Comparing the new "Neural Codec Language Model" approach (VALL-E) with the standard Regression approach.

|Feature|Traditional TTS (Tacotron 2 / FastSpeech 2)|VALL-E (Neural Codec Language Model)|
|---|---|---|
|**Core Task**|**Continuous Regression**: Predicts continuous Mel-spectrogram values.|**Language Modeling**: Predicts discrete acoustic codec codes (tokens) just like text.|
|**Output Representation**|**Mel-Spectrograms**: Continuous float values representing frequency energy.|**Discrete Codes**: Integers (indices) from a VQ-VAE codebook (e.g., EnCodec).|
|**Architecture**|**Encoder-Decoder**: Often uses LSTM or Transformer with explicit duration modeling.|**Decoder-Only Transformer**: Similar to GPT-3, uses causal masking.|
|**Zero-Shot Capability**|**Limited**: Requires fine-tuning or complex speaker encoders for new voices.|**Strong**: Can clone a voice from a 3-second prompt via "In-Context Learning".|
|**Training Data Scale**|**Small/Medium**: Typically trained on <1000 hours (e.g., LJSpeech).|**Massive**: Trained on 60,000+ hours (e.g., LibriLight) to learn general acoustics.|
|**Synthesis Method**|**Deterministic**: Usually produces the same output for the same text (unless explicitly varied).|**Probabilistic/Sampling**: Can generate diverse outputs (different prosody/tone) for the same text.|

## 9. Audio Representation: Continuous vs. Discrete

A fundamental shift in how audio is handled in modern Large Audio Models (like VALL-E).

|Feature|Continuous Representation (Mel-Spectrogram)|Discrete Representation (Audio Codec Codes)|
|---|---|---|
|**Data Type**|Floating point numbers (e.g., -4.52, 12.30).|Integers / Tokens (e.g., Token ID #402, #899).|
|**Loss Function**|**L1 / L2 Loss** (Mean Squared Error) to minimize distance.|**Cross-Entropy Loss** (Classification) to predict the next token probability.|
|**Compression**|Good for visualization, but harder for LMs to model directly.|Highly compressed; allows using standard LLM architectures (GPT) on audio.|
|**Resolution**|High frequency resolution (captures fine spectral details).|Quantized resolution (depends on the Codebook size, e.g., 1024 codes).|
|**Used In**|Tacotron, FastSpeech, WaveGlow.|VALL-E, AudioLM, MusicLM.|

## 10. Zero-Shot Learning Approaches in TTS

How different models handle "Unseen Speakers" (Voice Cloning).

|Method|Speaker Embedding (e.g., SV2TTS / YourTTS)|In-Context Learning (VALL-E)|
|---|---|---|
|**Mechanism**|Encodes a reference audio into a fixed-size vector ("d-vector") and conditions the decoder on it.|Concatenates the "Acoustic Prompt" tokens with the "Target Text" tokens as a single sequence.|
|**Analogy**|Like giving a painter a color palette (the vector) to paint a new picture.|Like giving a writer the first paragraph of a story (the prompt) and asking them to continue the style.|
|**Pros**|Fast inference; stable control over speaker identity.|Highly natural prosody; preserves acoustic environment (e.g., background noise).|
|**Cons**|Often sounds "averaged" or loses emotion; struggles with very short prompts.|Computationally expensive; potential stability issues (can babble or loop).|

## 11. VALL-E Architecture: AR vs. NAR Modeling

VALL-E splits the generation process into two distinct stages to handle the hierarchical structure of audio codes.

|Stage|Autoregressive (AR) Stage|Non-Autoregressive (NAR) Stage|
|---|---|---|
|**Code Layer**|Predicts the **1st Quantizer** code (the most dominant acoustic information).|Predicts **Quantizers 2 through 8** (fine acoustic details/residual errors).|
|**Conditioning**|Conditioned on **Text** tokens and **Prompt** acoustic tokens.|Conditioned on **Text**, **Prompt**, and the **1st Layer Code** (predicted in AR stage).|
|**Dependency**|Sequential: $Token_t$ depends on $Token_{t-1}$. Slow generation.|Parallel: All tokens in layers 2-8 are generated in parallel (usually).|
|**Goal**|Establish the content, prosody, and general speaker identity.|Refine the audio quality and remove artifacts/noise.|
|**Speed**|**Slow**: Linear time complexity with sequence length.|**Fast**: Constant time (or iterative refinement) regardless of length.|

## 12. Audio Codecs: EnCodec vs. SoundStream

VALL-E relies heavily on EnCodec. Understanding how it differs from other codecs is useful.

|Feature|EnCodec (Meta)|SoundStream (Google)|
|---|---|---|
|**Core Architecture**|Convolutional Encoder-Decoder + Residual Vector Quantization (RVQ).|Convolutional Encoder-Decoder + Residual Vector Quantization (RVQ).|
|**Discriminator**|Uses **MS-STFTD** (Multi-Scale Short-Time Fourier Transform Discriminator) for high fidelity.|Uses a mix of waveform and spectral discriminators.|
|**Key Innovation**|Optimized for **LM-based generation**; focuses on highly compressed discrete representations.|Optimized for **real-time communication** and varying bitrates.|
|**Usage**|Used in **VALL-E**, **MusicGen**.|Used in **AudioLM**, **Lyra**.|
|**Output**|8 parallel streams of discrete codes (at 24kHz).|Multiple streams of codes (at various sampling rates).|

## 13. System Comparison: VALL-E vs. AudioLM

Two major "Speech-from-Text" systems released around the same time.

|Feature|VALL-E (Microsoft)|AudioLM (Google)|
|---|---|---|
|**Input**|Phonemes (Text) + Acoustic Prompt.|Semantic Tokens (from w2v-BERT) + Acoustic Tokens.|
|**Intermediate Step**|**Directly maps** Text $\rightarrow$ Acoustic Tokens (EnCodec).|**Two-stage**: Text $\rightarrow$ Semantic Tokens $\rightarrow$ Acoustic Tokens (SoundStream).|
|**Semantic Understanding**|Implicitly learned by the model from massive data.|Explicitly modeled using **w2v-BERT** (captures meaning/phonetics separately).|
|**Consistency**|Can sometimes struggle with "content" (skipping/repeating words) due to no explicit alignment.|**High consistency** due to the intermediate semantic token stage (separates content from speaker style).|
|**Focus**|**Text-to-Speech (TTS)** and Voice Cloning.|**Speech Continuation** and Audio Generation (can do TTS but designed for continuation).|

## 14. Physics of Sound: Harmonics vs. Formants (Lecture 1)

A crucial distinction in the "Source-Filter" theory of speech production.

|Feature|Harmonics (Source)|Formants (Filter)|
|---|---|---|
|**Origin**|Created by the **Vocal Folds** (Cords) vibration.|Created by the **Vocal Tract** (Tube) shape (tongue/lip position).|
|**Frequencies**|Exact integer multiples of the Fundamental Frequency ($f_0$). (e.g., $100, 200, 300$ Hz).|Resonant frequencies of the air in the vocal tract. (e.g., $500, 1500, 2500$ Hz).|
|**Control**|Controlled by pitch/intonation (tightening vocal folds).|Controlled by articulation (moving tongue, jaw, lips).|
|**Perception**|Perceived as **Pitch**.|Perceived as **Vowel Quality** (e.g., /a/ vs /i/).|
|**Visual (Wideband)**|Visible as vertical striations (glottal pulses).|Visible as dark horizontal bands of energy.|

## 15. HMM Algorithms: Forward vs. Viterbi (Lecture 6 & 7)

Two core algorithms used in Hidden Markov Models with very different goals.

|Feature|Forward Algorithm|Viterbi Algorithm|
|---|---|---|
|**Goal**|**Evaluation**: Compute the likelihood of an observation sequence $P(O \mid \lambda)$.|**Decoding**: Find the single best hidden state sequence $Q$ that generated $O$.|
|**Operation**|**Summation (**$\sum$**)**: Sums probabilities of _all_ possible paths reaching a state.|**Maximization (**$\max$**)**: Keeps only the probability of the _best_ path reaching a state.|
|**Complexity**|$O(N^2 T)$|$O(N^2 T)$|
|**Backtracking?**|**No**. Just outputs a final probability score.|**Yes**. Must backtrack from the end to reconstruct the state sequence.|
|**Soft vs. Hard**|Soft alignment (probability mass).|Hard alignment (one specific path).|

## 16. Decoding Strategies: Greedy vs. Beam Search (Lecture 9)

How the model chooses the output text during inference.

|Feature|Greedy Decoding|Beam Search|
|---|---|---|
|**Decision Rule**|Picks the single **highest probability** token at each time step.|Maintains the top $k$ (beam size) most likely sequence hypotheses at each step.|
|**Memory**|Low. Only remembers the current state.|Higher. Must store $k$ active paths and their scores.|
|**Optimality**|**Local Optimum**: Can get stuck in a wrong path if a high-prob token leads to a dead end.|**Approximate Global Optimum**: Looks ahead to find better overall sequences.|
|**Risk**|Prone to "garden path" errors (cannot recover from a mistake).|Can recover from temporary low-probability transitions if the overall sentence score is high.|
|**Equation**|$\hat{y}_t = \underset{y \in V}{\operatorname{argmax}} P(y \mid y_{<t}, X)$|$S(Y) = \frac{1}{\vert Y \vert} (\log P(Y \mid X) + \lambda \log P_{LM}(Y))$|

## 17. Streaming ASR: Attention (AED) vs. Transducer (RNN-T) (Lecture 9)

Comparing standard global attention models with streaming-capable architectures.

|Feature|Attention-Based Encoder-Decoder (AED)|RNN-Transducer (RNN-T)|
|---|---|---|
|**Latency**|**High**: Typically processes the entire utterance at once (offline).|**Low**: Processes audio frame-by-frame (streaming/online).|
|**Dependencies**|**Global**: Cross-attention mechanism looks at _all_ input frames simultaneously.|**Local**: Prediction depends only on current audio frame + previous output tokens.|
|**Components**|Encoder + Decoder (with Cross-Attention).|Encoder + Prediction Network + Joint Network.|
|**Alignment**|**Soft**: Learned implicitly via attention weights.|**Harder**: Explicitly handles alignment via the lattice (similar to CTC but autoregressive).|
|**Use Case**|Whisper, Google Voice Search (Offline mode).|Real-time Captioning, Siri/Assistant (Streaming mode).|

## 18. Windowing: Rectangular vs. Hamming (Lecture 2/3)

A key signal processing choice to minimize spectral leakage (artifacts from cutting the signal).

|Feature|Rectangular Window|Hamming Window|
|---|---|---|
|**Shape**|Box shape (1 inside window, 0 outside). Sharp cut-off.|Bell/Cosine shape. Tapers the signal to zero at the edges.|
|**Spectral Leakage**|**High**: Creates large "side lobes" (noise) in the frequency domain.|**Low**: Suppresses side lobes significantly.|
|**Resolution**|**Narrow Main Lobe**: Good frequency resolution but noisy.|**Wider Main Lobe**: Slightly worse resolution but cleaner spectrum.|
|**Equation**|$w[n] = 1$|$w[n] = 0.54 - 0.46 \cos(\frac{2\pi n}{L-1})$|
|**Use Case**|Rarely used for speech; okay for transients.|**Standard for Speech**: Used in almost all MFCC/Spectrogram calculations.|

## 19. GMM Covariance Types: Diagonal vs. Full (Lecture 4)

A trade-off in Gaussian Mixture Models between model power and computational cost.

|Feature|Diagonal Covariance|Full Covariance|
|---|---|---|
|**Assumption**|Features are **independent** (uncorrelated).|Features are **correlated** (dependent).|
|**Parameters**|**Few**: $D$ variances per Gaussian ($O(D)$).|**Many**: $D \times D$ matrix per Gaussian ($O(D^2)$).|
|**Computation**|**Fast**: Inverting a diagonal matrix is trivial ($1/\sigma^2$).|**Slow**: Matrix inversion is computationally expensive ($O(D^3)$).|
|**Data Need**|**Low**: Requires less data to train robustly.|**High**: Prone to overfitting without massive data.|
|**Standard Practice**|**Preferred**: We decorrelate features (using DCT in MFCC) so we can use this simple model.|**Rare**: Only used if features are strongly correlated and data is abundant.|

## 20. Loss Functions: CTC Loss vs. Cross-Entropy (Lecture 8)

The core difference in how we train end-to-end ASR models.

|Feature|CTC Loss (Connectionist Temporal Classification)|Cross-Entropy (CE) Loss|
|---|---|---|
|**Requirement**|**Alignment-Free**: Does NOT need to know which frame maps to which phoneme.|**Aligned**: Requires explicit frame-to-phoneme alignment (or attention).|
|**Mechanism**|Sums probability over **all valid paths** that yield the target text.|Maximizes probability of the **correct token** at each specific time step.|
|**Blank Token**|**Crucial**: Uses a "blank" ($\epsilon$) to handle silence and repeated characters.|**Not needed** (unless used in specific architectures like RNN-T).|
|**Assumption**|**Conditional Independence**: Output at time $t$ is independent of $t-1$ (given input).|**Dependent**: In autoregressive models, output depends on history $y_{<t}$.|
|**Use Case**|Encoder-only models (QuartzNet), ASR pre-training.|Autoregressive Decoders (Whisper, LAS, VALL-E).|

## 21. CNN Architectures: 1D vs. 2D Convolution (Lecture 9)

Different ways to process audio spectrograms in Deep Learning.

|Feature|1D Convolution|2D Convolution|
|---|---|---|
|**Input View**|Treats input as a **Time Series** with $F$ channels (features).|Treats input as an **Image** (Time $\times$ Frequency).|
|**Kernel Movement**|Slides only along the **Time** axis.|Slides along both **Time** and **Frequency** axes.|
|**Feature Learning**|Learns temporal patterns (e.g., "onset of a sound").|Learns spectral-temporal patterns (e.g., "formant slope").|
|**Invariance**|Translation invariant in **Time** only.|Translation invariant in **Time and Frequency** (good for pitch shifts).|
|**Typical Use**|Raw waveform inputs (Wav2Vec) or simple TDNNs.|Spectrogram inputs (DeepSpeech 2, VGG-ish models).|

## 22. Language Models: N-gram vs. Neural LM (Lecture 9)

Two generations of probability models used to correct ASR output.

|Feature|N-gram Model (Statistical)|Neural Language Model (RNN/Transformer)|
|---|---|---|
|**History**|**Short / Fixed**: Looks only at $N-1$ previous words (e.g., 2 or 3).|**Long / Infinite**: Can theoretically look back at the entire sentence history.|
|**Representation**|**Sparse**: counts word sequences exactly as seen in training.|**Dense**: Uses word embeddings; learns similarity between words (e.g., "cat" $\approx$ "dog").|
|**Generalization**|**Poor**: Assigns zero probability to unseen sequences (requires smoothing).|**Good**: Can generalize to unseen sequences based on semantic similarity.|
|**Size**|**Huge RAM**: Grows exponentially with $N$ (billions of counts).|**Compact**: Fixed model size (weights), though compute-heavy.|
|**Use In ASR**|Fast initial decoding (First Pass).|Rescoring hypotheses (Second Pass) or directly inside E2E models.|