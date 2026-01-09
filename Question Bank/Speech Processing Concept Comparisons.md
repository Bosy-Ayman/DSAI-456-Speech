
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
|**Information Loss**|**Higher**: DCT discards some information (higher "quefrencies") to compress the signal into 13-39 coefficients.|**Lower**: Retains more physical detail of the sound, acting as a high-resolution "image" of the audio.|
|**Visualization**|Abstract and hard for humans to interpret visually (cepstral domain).|Easy to interpret; looks like a "heatmap" of sound over time where axes are Time vs. Frequency.|

## 3. Feature Dynamics: Static vs. Delta Features

Based on Lecture 2/3, raw spectral features often fail to capture the _movement_ of speech.

|Feature|Static Features|Dynamic Features (Deltas $\Delta$ & Delta-Deltas $\Delta\Delta$)|
|---|---|---|
|**Definition**|The raw values of the feature vector (e.g., 13 MFCCs) at a specific time frame $t$.|The **first and second derivatives** of the static features over time (velocity and acceleration).|
|**Purpose**|Describes the **spectral shape** (vocal tract configuration) at a single instant.|Describes **how speech changes** (transitions between phonemes, stopping/starting).|
|**Calculation**|Computed directly from the windowed signal frame.|Computed using a regression formula over neighbor frames: $d_t = \frac{\sum (c_{t+n} - c_{t-n})}{2 \sum n^2}$.|
|**Dimensionality**|Usually 13 coefficients (for MFCC).|Adds 26 coefficients (13 Delta + 13 Delta-Delta), bringing total feature vector to 39.|
|**Necessity**|Crucial for identifying steady-state vowels.|Crucial for identifying **transients, stops, and diphthongs** where the sound is moving.|

## 4. Signal Analysis: Narrowband vs. Wideband Spectrograms

This comparison focuses on the trade-off between time and frequency resolution (Heisenberg uncertainty principle in signal processing).

|Feature|Narrowband Spectrogram|Wideband Spectrogram|
|---|---|---|
|**Window Length**|**Long** window (e.g., 20ms - 30ms). Good for capturing stable frequencies.|**Short** window (e.g., 3ms - 5ms). Good for capturing fast changes.|
|**Frequency Resolution**|**High**: Can distinguish individual **Harmonics** (multiples of $f_0$).|**Low**: Smears harmonics together; cannot resolve fine frequency differences.|
|**Time Resolution**|**Low**: Smears rapid changes in time; blurs transient events like stop bursts.|**High**: Can detect precise timing of events (like individual glottal pulses).|
|**Visual Structure**|Shows **horizontal striations** (harmonics). Good for tracking pitch contours.|Shows **vertical striations** (glottal pulses) and broad dark bands (**Formants**).|
|**Primary Use**|**Pitch (F0) tracking**, intonation analysis, tone languages.|**Formant analysis**, boundary detection, vowel classification.|

## 5. Vowel Formant Analysis: F1 vs. F2

Based on Lecture 1, formants are the resonant frequencies of the vocal tract and define vowel identity.

|Feature|First Formant (F1)|Second Formant (F2)|
|---|---|---|
|**Physical Correlate**|Inversely related to **Tongue Height** (Jaw opening).|Related to **Tongue Advancement** (Front vs. Back position).|
|**Rule of Thumb**|**High F1** = Low Tongue (Open mouth) e.g., /a/ "bat".<br><br>**Low F1** = High Tongue (Closed mouth) e.g., /i/ "beet".|**High F2** = Front Tongue e.g., /i/ "beet".<br><br>**Low F2** = Back Tongue e.g., /u/ "boot".|
|**Frequency Range**|Typically **200 Hz – 1000 Hz**.|Typically **800 Hz – 2500 Hz**.|
|**Perception**|Distinguishes "open" vs "close" vowels.|Distinguishes "front" vs "back" vowels.|

## 6. End-to-End ASR: CTC vs. Attention (LAS)

Comparison of architectures that replaced the traditional HMM-DNN hybrid approach.

|Feature|CTC (Connectionist Temporal Classification)|Attention-Based (e.g., LAS, Transformer)|
|---|---|---|
|**Alignment**|**Hard/Implicit**: Uses a "blank" token and dynamic programming to handle alignment. Sums over all valid paths.|**Soft/Explicit**: Uses an attention mechanism (Cross-Attention) to compute a weighted sum of input frames for each output step.|
|**Independence**|**Conditional Independence**: Assumes outputs are independent given the input. Can predict "A A" as "A" without a blank separator.|**Auto-regressive**: Prediction at step $t$ depends on predictions at $t-1$. implicitly learns a Language Model.|
|**Monotonicity**|Strictly **monotonic** (speech naturally moves forward in time).|Can be **non-monotonic** (potentially problematic for speech; requires constraints to prevent looping or skipping).|
|**Streaming**|Easier to adapt for **streaming/online** recognition (output can be generated as soon as input arrives).|Harder to stream; Global attention typically requires seeing the **entire utterance** before outputting the first token.|
|**Weakness**|struggle with language semantics (e.g., spelling) without an external Language Model.|Prone to **hallucinations** or infinite loops if the attention mechanism fails to advance.|

## 7. Metric: WER vs. CER & Error Types

Common metrics for evaluating Speech Recognition systems.

|Metric|WER (Word Error Rate)|CER (Character Error Rate)|
|---|---|---|
|**Unit of Error**|Whole **Words**.|Individual **Characters** (letters).|
|**Formula**|$(S + D + I) / N_{words}$|$(S + D + I) / N_{chars}$|
|**Difficulty**|Generally **higher** (harder). A single wrong letter makes the whole word wrong ("cat" $\to$ "bat" is 100% error).|Generally **lower**. "cat" $\to$ "bat" is only 33% error (1 char wrong).|
|**Use Case**|Standard for general speech recognition (Siri, Google Assistant).|Used for languages with complex scripts (Chinese, Japanese) or agglutinative languages.|

**Types of ASR Errors (from Lecture 9):**

- **Substitution (S):** System replaces a word with another (e.g., "cat" $\to$ "cap").
    
- **Deletion (D):** System omits a word (e.g., "a big cat" $\to$ "a cat").
    
- **Insertion (I):** System adds a word not in the reference (e.g., "cat" $\to$ "the cat").
    

## 8. The Three Fundamental HMM Problems

As outlined in Lectures 5, 6, and 7, HMMs are defined by solving three specific problems.

|Problem|Goal|Algorithm|Complexity|
|---|---|---|---|
|**1. Evaluation**|Compute the likelihood of an observation sequence given a model: $P(O \mid \lambda)$.|**Forward Algorithm** (sums probabilities of all paths).|$O(N^2 T)$|
|**2. Decoding**|Find the single most likely sequence of hidden states $Q$ that generated the observations.|**Viterbi Algorithm** (maximizes probability using dynamic programming).|$O(N^2 T)$|
|**3. Learning**|Estimate the model parameters $\lambda = (A, B, \pi)$ to maximize the likelihood of the training data.|**Baum-Welch Algorithm** (Expectation-Maximization / EM).|Iterative $O(N^2 T)$|

## 9. Model Selection: AIC vs. BIC

From Lecture 4, used when deciding the number of components in a GMM (or parameters in a model) to prevent overfitting.

|Feature|AIC (Akaike Information Criterion)|BIC (Bayesian Information Criterion)|
|---|---|---|
|**Formula**|$-2 \ln(L) + 2k$|$-2 \ln(L) + k \ln(N)$|
|**Terms**|$L$: Likelihood, $k$: # parameters.|$L$: Likelihood, $k$: # parameters, $N$: Sample size.|
|**Penalty**|Penalizes complexity **linearly** with $k$.|Penalizes complexity **logarithmically** with data size $N$.|
|**Behavior**|Tends to select more complex models (larger $k$) when $N$ is large.|**Stricter**: Penalizes complexity more heavily. Selects simpler models (sparse).|
|**Use Case**|Better for prediction tasks.|Better for consistency (finding the "true" model).|

## 10. TTS Paradigm: VALL-E vs. Traditional

Comparing the new "Neural Codec Language Model" approach with standard Regression.

|Feature|Traditional TTS (Tacotron 2 / FastSpeech 2)|VALL-E (Neural Codec Language Model)|
|---|---|---|
|**Core Task**|**Continuous Regression**: Predicts continuous Mel-spectrogram values (float numbers).|**Language Modeling**: Predicts discrete acoustic codec codes (integers) just like text tokens.|
|**Output Representation**|**Mel-Spectrograms**: Requires a Vocoder (e.g., HiFi-GAN) to convert Mels to Audio.|**Discrete Codes**: Integers from a VQ-VAE codebook (e.g., EnCodec) converted directly to audio.|
|**Zero-Shot Capability**|**Limited**: Requires complex x-vector/d-vector speaker encoders to clone voices.|**Strong**: Uses "In-Context Learning". Cloning is done by simply prepending a 3-second audio prompt.|
|**Training Data Scale**|**Small/Medium**: Typically <1,000 hours (e.g., LJSpeech).|**Massive**: Trained on 60,000+ hours (LibriLight) to learn general acoustics.|
|**Synthesis Method**|**Deterministic**: Usually produces the same output for the same text.|**Probabilistic**: Uses sampling (Top-k / Nucleus), allowing for diverse prosody for the same text.|

## 11. VALL-E Architecture: AR vs. NAR Modeling

VALL-E splits generation into two stages to handle the 8-layer hierarchical codebook of EnCodec.

|Stage|Autoregressive (AR) Stage|Non-Autoregressive (NAR) Stage|
|---|---|---|
|**Target**|Predicts the **1st Quantizer** code (Layer 1).|Predicts **Quantizers 2 through 8** (Layers 2-8).|
|**Role**|Captures the **content**, prosody, and general speaker identity (coarse information).|Refines the audio quality and removes quantization artifacts/noise (fine details).|
|**Dependency**|Sequential: $Token_t$ depends on $Token_{t-1}$.|Parallel: All tokens in layers 2-8 are generated in parallel (or iteratively per layer).|
|**Conditioning**|Conditioned on **Text** tokens + **Acoustic Prompt**.|Conditioned on **Text**, **Prompt**, and the **1st Layer Code** (from AR stage).|
|**Speed**|**Slow**: Time grows linearly with sequence length.|**Fast**: Constant time (relative to sequence length) as layers are predicted in parallel.|

## 12. Audio Codecs: EnCodec vs. SoundStream

VALL-E relies on EnCodec. Comparing it with Google's SoundStream (used in AudioLM).

|Feature|EnCodec (Meta)|SoundStream (Google)|
|---|---|---|
|**Core Architecture**|Convolutional Encoder-Decoder + Residual Vector Quantization (RVQ).|Convolutional Encoder-Decoder + Residual Vector Quantization (RVQ).|
|**Optimization**|Optimized for **Generative Modeling**; high compression to fit token limits of LLMs.|Optimized for **Real-Time Communication** (low latency) and varying bitrates.|
|**Discriminator**|Uses **MS-STFTD** (Multi-Scale Short-Time Fourier Transform) for spectral fidelity.|Uses a mix of waveform-based and spectral-based discriminators.|
|**Structure**|Produces 8 parallel streams of discrete codes at 24kHz.|Produces multiple streams, often focusing on semantic vs. acoustic separation.|

## 13. System Comparison: VALL-E vs. AudioLM

Two major "Speech-from-Text" systems.

|Feature|VALL-E (Microsoft)|AudioLM (Google)|
|---|---|---|
|**Input**|Phonemes (Text) + Acoustic Prompt.|Semantic Tokens (w2v-BERT) + Acoustic Tokens.|
|**Mapping Strategy**|**Direct Map**: Text $\rightarrow$ Acoustic Tokens (EnCodec).|**Two-Stage**: Text $\rightarrow$ Semantic Tokens $\rightarrow$ Acoustic Tokens.|
|**Semantic Control**|Implicitly learned from massive data.|Explicitly modeled using **w2v-BERT** (separates meaning from speaker style).|
|**Consistency**|Can struggle with content (skipping/repeating words) due to no explicit alignment.|High consistency due to the intermediate semantic token stage.|
|**Primary Focus**|**Text-to-Speech (TTS)** and Zero-Shot Voice Cloning.|**Speech Continuation** (completing audio) and Generation.|

## 14. HMM Algorithms: Forward vs. Viterbi

Detailed comparison of the algorithms used in Lecture 6 & 7.

|Feature|Forward Algorithm|Viterbi Algorithm|
|---|---|---|
|**Goal**|**Evaluation**: Calculates the total probability of an observation sequence $P(O)$.|**Decoding**: Finds the single best path (state sequence) $Q$ that explains $O$.|
|**Mathematical Op**|**Summation (**$\sum$**)**: "Soft" accumulation of probabilities from all possible paths.|**Maximization (**$\max$**)**: "Hard" selection of the single best predecessor.|
|**Backtracking?**|**No**. It outputs a single probability value at the end.|**Yes**. Essential to reconstruct the sequence of states from end to start.|
|**Output**|A probability score (Likelihood).|A sequence of states (e.g., Phoneme sequence).|

## 15. Decoding Strategies: Greedy vs. Beam Search

From Lecture 9, how models choose the output text during inference.

|Feature|Greedy Decoding|Beam Search|
|---|---|---|
|**Mechanism**|Picks the single **highest probability** token at each time step ($t$).|Maintains a set of $k$ (beam width) most likely hypotheses at each step.|
|**Memory Usage**|**Low**: Only stores the current state.|**Medium/High**: Must store $k$ active paths and their cumulative scores.|
|**Optimality**|**Local Optimum**: Can get stuck in a "garden path" (a mistake early on ruins the sentence).|**Approximate Global Optimum**: Can recover from a temporary low-probability transition if the overall sentence is better.|
|**Formula**|$\hat{y}_t = \arg\max P(y_t \mid y_{<t}, X)$|Maximizes Score $S(Y) = \log P(Y|

## 16. Streaming ASR: Attention (AED) vs. Transducer (RNN-T)

|Feature|Attention-Based Encoder-Decoder (AED)|RNN-Transducer (RNN-T)|
|---|---|---|
|**Latency**|**High**: Typically processes the entire utterance (Offline).|**Low**: Processes audio frame-by-frame (Streaming).|
|**Dependencies**|**Global**: Cross-attention looks at the _entire_ input sequence for every output token.|**Local**: Prediction depends on the current audio encoder frame + previous output tokens.|
|**Alignment**|**Soft**: Learned implicitly via attention weights.|**Monotonic**: Explicitly enforced by the lattice structure (similar to CTC).|
|**Use Case**|Batch processing, Offline Voice Search, Subtitling.|Real-time Captioning, Live Dictation, Smart Speakers.|

## 17. Windowing: Rectangular vs. Hamming

From Lecture 2, minimizing spectral leakage.

|Feature|Rectangular Window|Hamming Window|
|---|---|---|
|**Shape**|Box shape (1 inside window, 0 outside). Sudden cut-off at edges.|Bell/Cosine shape. Tapers the signal gently to zero at edges.|
|**Spectral Leakage**|**High**: Sharp edges cause discontinuities, creating artificial "side lobes" in the spectrum.|**Low**: Tapering reduces discontinuities, suppressing side lobes.|
|**Main Lobe Width**|**Narrow**: Good frequency resolution but noisy surroundings.|**Wider**: Slightly lower frequency resolution but much cleaner spectrum.|
|**Equation**|$w[n] = 1$|$w[n] = 0.54 - 0.46 \cos(\frac{2\pi n}{L-1})$|

## 18. GMM Covariance Types: Diagonal vs. Full

From Lecture 4, balancing model complexity.

|Feature|Diagonal Covariance|Full Covariance|
|---|---|---|
|**Assumption**|Features are **independent** (uncorrelated). Dimensions do not affect each other.|Features are **correlated**. Modeling the relationship between all dimensions.|
|**Parameters**|**Few**: $D$ variances per Gaussian ($O(D)$).|**Many**: $D \times D$ covariance matrix per Gaussian ($O(D^2)$).|
|**Computation**|**Fast**: Inversion is trivial ($1/\sigma^2$).|**Slow**: Matrix inversion is computationally expensive ($O(D^3)$).|
|**Requirement**|Requires **Decorrelated Features** (like MFCCs).|Can handle **Correlated Features** (like Spectrograms).|

## 19. Loss Functions: CTC vs. Cross-Entropy

From Lecture 8 and 9.

|Feature|CTC Loss (Connectionist Temporal Classification)|Cross-Entropy (CE) Loss|
|---|---|---|
|**Alignment**|**Alignment-Free**: Sums over all possible alignments between input $X$ and text $Y$.|**Aligned**: Requires explicit frame-to-token alignment (or uses Attention to align).|
|**Blank Token**|**Required**: Uses $\epsilon$ to represent silence or boundaries between repeated characters.|**Not Used** (Standard CE): Predicts tokens directly.|
|**Independence**|**Conditional Independence**: Assumes output at $t$ depends only on input (not previous output).|**Dependent**: In autoregressive models, loss depends on previous ground-truth tokens.|
|**Application**|Encoder-Only models (QuartzNet, Wav2Vec2).|Encoder-Decoder models (Whisper, LAS).|

## 20. CNN Architectures: 1D vs. 2D Convolution

From Lecture 9.

|Feature|1D Convolution|2D Convolution|
|---|---|---|
|**Input View**|Treats input as a **Time Series** with $F$ feature channels.|Treats input as an **Image** with dimensions Time $\times$ Frequency.|
|**Kernel Movement**|Slides only along the **Time** axis.|Slides along both **Time** and **Frequency** axes.|
|**Invariance**|Translation invariant in **Time** only.|Translation invariant in **Time and Frequency** (robust to slight pitch shifts).|
|**Typical Use**|Processing raw waveforms or simple feature extraction.|Processing Spectrograms where spectral patterns (like formants) shift.|