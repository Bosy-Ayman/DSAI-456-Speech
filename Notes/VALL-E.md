## Part 1: What is the Audio Codec (EnCodec)?

### The Problem It Solves

Audio is stored as 16-bit numbers (0-65,536 possible values per sample). At 24,000 Hz sampling rate, a 10-second audio clip has 240,000 samples. A language model would need to predict from 65,536 possible values at each timestep—computationally impossible.

### How EnCodec Works

EnCodec is a neural encoder-decoder that compresses audio using **Residual Vector Quantization (RVQ)**.

**Step 1: Encoding**

- Input: 10 seconds of 24 kHz audio = 240,000 samples
- Encoder downsamples this to 750 timesteps (24,000 × 10 / 320 = 750)
- This is a 320× reduction in sequence length

**Step 2: Quantization (8 levels)** EnCodec uses 8 "quantizers" arranged hierarchically:

```
Timestep t | Q1   Q2   Q3   Q4   Q5   Q6   Q7   Q8
-----------|--------------------------------------
t=0        | 342  127  45   89   12   201  56   34
t=1        | 401  203  78   156  34   145  89   67
t=2        | 288  95   52   71   6    178  23   45
...        | ...
t=750      | 521  412  301  267  189  234  156  98
```

**What each quantizer captures:**

- Q1 (first): Speaker identity, pitch, gross acoustic properties (~30% of quality)
- Q2-Q8 (later): Fine details like breath, texture, emotion (~70% of quality)

**Decoding** The decoder reconstructs audio from these 750 × 8 = 6,000 tokens instead of 240,000 samples.

### Why This Matters for VALL-E

VALL-E doesn't predict raw audio waveforms. Instead, it predicts these **8 sequences of tokens**, one at a time.

---

## Part 2: The Two-Stage Language Modeling

### Stage 1: Autoregressive (AR) Model for Q1 Only

**Input at training time:**

- Phoneme sequence: `[p, ə, n, m, ə]` (from word "panima")
- 3-second speaker voice converted to codes: First quantizer codes `[342, 401, 288, ...]`
- Target codes to predict: First quantizer `[521, 503, 488, ...]`

**What the model does:** Uses a standard transformer decoder to predict tokens **one at a time, left to right**:

```
Position 0: Given [phonemes] + [speaker codes], predict code 521
Position 1: Given [phonemes] + [speaker codes] + [521], predict code 503
Position 2: Given [phonemes] + [speaker codes] + [521, 503], predict code 488
...
```

**Equations from the paper (simplified):**

```
p(c₀,₁|x, C̃₀,₁) = probability of predicting 521 given phonemes and speaker
p(c₁,₁|c₀,₁, x, C̃₀,₁) = probability of predicting 503 given previous output
p(c₂,₁|c₀,₁, c₁,₁, x, C̃₀,₁) = probability of predicting 488 given all previous
```

**Why autoregressive for Q1?**

- Q1 determines sequence length (how long the speech will be)
- Each speaker speaks at different speeds
- Can't predict length in advance, so need autoregressive generation

**Transformer architecture used:**

- 12 layers
- 16 attention heads
- 1024 embedding dimension
- Uses causal masking (token can only attend to previous tokens)

### Stage 2: Non-Autoregressive (NAR) Models for Q2-Q8

**The problem with doing autoregressive for all 8 quantizers:**

- Would need 8 sequential passes, each predicting 750 tokens
- Extremely slow: O(T) = O(750) × 8 = 6,000 operations per sample

**NAR solution:** Predict ALL tokens in Q2 simultaneously (same transformer pass).

**Input at training time:**

- Phoneme sequence: `[p, ə, n, m, ə]`
- Speaker codes (all 8 quantizers summed): `[342+127+45+89+12+201+56+34, ...]`
- Q1 codes generated from AR model: `[521, 503, 488, ...]`
- Target: Generate all of Q2 simultaneously

**The key equation:**

```
For stage j ∈ [2, 8]:
  p(c:,j | c:,<j, x, C̃, j) = ∏ᵢ p(cᵢ,j | c:,<j, x, C̃, j)
```

Translation: Probability of all Q2 tokens = product of individual probabilities (assuming independence).

**How it works:**

```
Input embeddings:
- x (phonemes) can attend to all positions
- C̃ (speaker voice codes summed) can attend to all positions
- c:,<2 (Q1 codes) can attend to all positions
- Each output position attends to everything (no causal masking)

Output: All 750 Q2 codes predicted in parallel
```

**Adaptive Layer Normalization (AdaLN):** Because the NAR model predicts for multiple stages, they inject which stage (j ∈ [2,8]) with AdaLN:

```
Output = αⱼ × LayerNorm(hidden_state) + βⱼ
where αⱼ and βⱼ depend on stage embedding
```

---

## Part 3: Training Process

### Data Preparation

**Raw data:** LibriLight (60,000 hours of audiobook speech)

**Step 1: Transcription**

- These are unlabeled audiobooks
- Train a hybrid DNN-HMM ASR model on labeled LibriSpeech (960 hours)
- Use this to transcribe all 60K hours (imperfectly, but okay)
- Output: Phoneme-level alignments at 30ms frameshift

**Step 2: Codec Encoding**

- For each audio file, run EnCodec encoder
- Get 750 × 8 token matrices per 10 seconds

### AR Model Training

**For each 10-second clip:**

1. Randomly crop to 10-20 seconds (for data augmentation)
2. Input entire clip (phonemes + codes) to transformer
3. Predict next code token for each position
4. Loss = cross-entropy comparing predicted token to ground truth

**Formula:**

```
Loss_AR = -∑ₜ log p(cₜ,₁ | c₍ₜ, x, C̃)
```

### NAR Model Training

**For each 10-second clip:**

1. Randomly sample a stage i ∈ [2, 8]
2. Input: phonemes + speaker codes + codes from Q1 to Qi-1
3. Predict all codes in stage i simultaneously
4. Loss = cross-entropy for all positions in stage i

**Formula:**

```
Loss_NAR = -∑ᵢ₌₂⁸ ∑ₜ log p(cₜ,ᵢ | c:,<i, x, C̃, i)
```

**Why random stage sampling?** Saves computation: Training every stage every batch would require 8 forward passes.

### Training Details

- Batch size: 6,000 acoustic tokens per GPU
- 16 V100 GPUs
- 800,000 training steps
- Learning rate: warmup to 5×10⁻⁴, then linear decay
- Optimizer: AdamW

---

## Part 4: Inference (Zero-Shot TTS)

### Setup

**Given:**

- Text to synthesize: "Hello world"
- Speaker voice sample: 3 seconds of speaker saying something else

**Step 1: Prepare Prompts**

**Phoneme prompt:**

- Text "Hello world" → phonemes: `[h, ə, l, oʊ, w, ɝ, l, d]`
- Speaker sample text → get its phonemes too

**Acoustic prompt:**

- Take 3-second speaker sample
- Run through EnCodec: get matrix C̃ (size: ~225 × 8, since 3 sec = 225 timesteps)

### Step 2: AR Decoding

**Input to AR model:**

```
[phoneme_speaker_sample] + [Q1_codes_of_speaker_sample] + 
[phoneme_target_text]
```

**Decoding process (sampling-based):**

```
Position 0:
  - Model outputs probability distribution over 1024 Q1 values
  - Instead of picking highest probability (greedy), SAMPLE from distribution
  - Get token, say 521
  
Position 1:
  - Model sees [521] + context
  - Outputs probability distribution
  - Sample → get 503
  
... continue until <EOS> token
```

**Why sampling instead of greedy?**

- Beam search can cause infinite loops
- Sampling creates diversity (same input → different outputs)

**Output:** Sequence of Q1 tokens for target text, e.g., `[521, 503, 488, 445, 512, 498, 520, 501]`

### Step 3: NAR Decoding (7 times)

**For stage j = 2:**

```
Input: 
  - phoneme_target_text
  - C̃ (3-second speaker codes, all 8 quantizers)
  - c:,1 (Q1 codes from AR step)
  
Forward pass (one pass, all positions in parallel)
Output: All 750 positions of Q2 codes
```

**For stages j = 3 to 8:**

- Repeat the same process
- NAR model receives codes from all previous stages
- Outputs stage j codes

**Total time: 1 AR pass + 7 NAR passes** (much faster than 8 AR passes)

### Step 4: Decode to Waveform

**Input to EnCodec decoder:**

- 750 × 8 token matrix (Q1-Q8 codes)

**Output:**

- 24 kHz audio waveform (750 timesteps × 320 = 240,000 samples)

---

## Part 5: Why It Works So Well

### Key Insight 1: In-Context Learning

Traditional TTS needs fine-tuning to adapt to new speakers. VALL-E doesn't.

**Why?** The 3-second speaker sample is treated like a "prompt" in a language model. The model learns to condition its output on speaker characteristics in the prompt, just like GPT-3 learns to answer questions after seeing examples in the prompt.

**Evidence from ablation (Table 5):**

```
With acoustic prompt:     Speaker similarity = 0.585
Without acoustic prompt:  Speaker similarity = 0.236
Difference: 0.349 points (huge!)
```

### Key Insight 2: Massive Data Beats Complex Design

**Previous zero-shot TTS:**

- 600 hours of data max
- Complex speaker encoders
- Required fine-tuning
- Generalization failed badly

**VALL-E:**

- 60,000 hours (100× more)
- Simple architecture (just transformers)
- No fine-tuning needed
- Generalizes well

Scaling laws from language models apply to speech too.

### Key Insight 3: Discrete Tokens > Continuous Regression

**Old approach:**

```
Text → predict continuous mel-spectrogram values → convert to waveform
Problem: Predicting continuous values is harder than discrete classification
```

**VALL-E approach:**

```
Text → predict discrete tokens (1024 choices per timestep) → convert to waveform
Problem: Classification is easier than regression in deep learning
```

### Key Insight 4: Hierarchical Codes Match Codec Structure

EnCodec's RVQ naturally separates:

- Speaker identity (Q1)
- Acoustic details (Q2-Q8)

VALL-E mirrors this:

- AR model (speaker-determining stage)
- NAR models (detail stages)

This is more efficient than trying to predict all codes simultaneously.

---

## Part 6: Experimental Results

### Metrics Used

**Automatic metrics:**

1. **Speaker Similarity (SPK)**
    
    - Uses WavLM-TDNN (state-of-art speaker verification model)
    - Compares synthesized speech to original speaker's voice
    - Range: [-1, 1], higher is better
    - Ground truth = 0.754 on LibriSpeech
2. **Word Error Rate (WER)**
    
    - Run speech recognition on synthesized audio
    - Compare to original text
    - Lower is better
    - Ground truth = 2.2% on LibriSpeech

**Human evaluation:**

1. **SMOS (Similarity Mean Opinion Score)**
    
    - 1-5 scale: How similar is synthesized voice to speaker?
    - 6 human raters per sample
2. **CMOS (Comparative Mean Opinion Score)**
    
    - -3 to +3 scale: How does VALL-E compare to baseline?
    - 12 human raters per sample

### LibriSpeech Results (Table 2-3)

**Automatic (Table 2):**

```
              WER    SPK
Ground Truth  2.2%   0.754
YourTTS       7.7%   0.337
VALL-E        5.9%   0.580
VALL-E-cont   3.8%   0.508
```

Interpretation:

- VALL-E's WER is 25% lower than baseline (5.9 vs 7.7)
- VALL-E's speaker similarity is 72% of ground truth (0.580 vs 0.754)—excellent for zero-shot
- VALL-E-continual is even better because AR model gets actual first 3 seconds

**Human (Table 3):**

```
             SMOS          CMOS
YourTTS      3.45±0.09    -0.12
VALL-E       4.38±0.10     0.00
Ground Truth 4.50±0.10    +0.17
```

Interpretation:

- VALL-E gets 4.38/5.0 on naturalness—almost as good as real humans (4.50)
- VALL-E beats YourTTS by 0.93 points on speaker similarity
- VALL-E is basically as natural as real speech (CMOS near 0)

### VCTK Results (Table 6-7)

VCTK has 108 speakers with various accents.

**Key finding (Table 6):**

```
3-second prompt:
- YourTTS on full data: 0.357
- VALL-E (unseen speakers): 0.382
- Ground Truth: 0.546
```

Even though YourTTS saw 97 speakers in training, VALL-E—which saw none of them—performs better on unseen speakers.

---

## Part 7: Ablation Studies (What Makes It Work?)

### NAR Ablation (Table 4)

**Test:** Remove acoustic prompts progressively

```
Setting                  WER    SPK
NAR-no prompt           19.6    0.518
NAR-phn prompt only      3.0    0.541
NAR-2 prompts (our)      2.8    0.732
```

**Finding 1:** Without any prompt, WER is 19.6 (70% of words wrong!)

- Shows the model can't learn content without phoneme guidance

**Finding 2:** Phoneme prompt fixes content (WER: 19.6 → 3.0)

- Phonemes drive what words are spoken

**Finding 3:** Acoustic prompt fixes speaker (SPK: 0.541 → 0.732)

- Acoustic codes are critical for speaker identity

### AR Ablation (Table 5)

```
Setting                WER    SPK
VALL-E (full)          5.9    0.585
w/o acoustic prompt    5.9    0.236
```

**Finding:**

- Content (WER) doesn't change without acoustic prompt (both 5.9)
- But speaker similarity collapses (0.585 → 0.236)
- Means AR model MUST see speaker codes to clone voice

---

## Part 8: Qualitative Analysis

### Diversity

VALL-E samples tokens (random), so same input → different outputs.

**Example (Figure 4a):** Same text synthesized twice:

1. First: faster speech rate
2. Second: slower speech rate, different phrasings

Previous systems deterministic—same input always gives identical output.

### Acoustic Environment Preservation

If speaker sample has reverberation (echo/room reflection), synthesized speech also has reverberation.

**Why?** EnCodec preserves fine acoustic details in Q2-Q8 codes. Large diverse training data (60K hours) includes many acoustic environments.

### Emotion Preservation

Take a 3-second sample where speaker sounds angry. Synthesized speech in angry emotion.

**Why?** Emotion is encoded in acoustic features (pitch contour, intensity) that appear in Q2-Q8 codes. Model learns to copy emotion from acoustic prompt.

---

## Part 9: Known Limitations

### 1. Robustness Issues

Some words deleted, inserted, or duplicated.

**Cause:** AR model attention misalignment (same problem as vanilla Transformer TTS)

**Fix proposed:** Use non-autoregressive models or modify attention (future work)

### 2. Data Coverage

- 60K hours still doesn't cover all accents (worse on VCTK than LibriSpeech)
- LibriLight is audiobooks (mostly reading style, not conversational)
- Limited prosody diversity

**Solution:** Scale to more data (they suggest this will "almost solve" zero-shot TTS)

### 3. Model Structure

- Uses two separate models (AR + NAR)
- Could potentially use one unified model
- Could make NAR fully parallel (currently 7 sequential NAR passes)

### 4. Misuse Risk

Can synthesize convincing voices → potential for voice spoofing/impersonation

**Mitigation:** Detection models, responsible deployment

---

## Summary Table: Why VALL-E Works

|Aspect|Before|VALL-E|Improvement|
|---|---|---|---|
|Training Data|600 hours|60,000 hours|100×|
|Architecture|Custom speaker encoder|Standard transformers|Simpler|
|Fine-tuning|Required|Not required|In-context learning|
|Intermediate Rep|Mel-spectrogram (continuous)|Discrete codes|Language modeling trick|
|Speaker Similarity|0.337|0.580|72% of human level|
|Naturalness vs. Human|-0.12 CMOS|+0.04 CMOS|Matches humans on VCTK|

The key insight: **Scaling + discrete tokens + hierarchical generation = better zero-shot speech synthesis**