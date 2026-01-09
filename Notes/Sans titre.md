# Study Summary: Automatic Speech Recognition (ASR)

This summary covers the fundamental concepts, architectures, and evaluation metrics for Automatic Speech Recognition (ASR) as presented in Chapter 15 of "Speech and Language Processing" by Jurafsky & Martin.

---

## 1. Introduction to ASR
**Automatic Speech Recognition (ASR)** is the task of mapping an acoustic waveform to its corresponding string of words.

### Dimensions of Variation
The difficulty of ASR depends on several factors:
*   **Vocabulary Size:** Small (e.g., 2-word "yes/no" or 11-word digits) vs. Large (60,000+ words).
*   **Speaker Type:** Dictation/Read speech (easier) vs. Conversational speech (hardest).
*   **Channel & Noise:** Quiet room with head-mounted mic (easier) vs. Noisy street with distant mic (harder).
*   **Speaker Characteristics:** Standard dialects (easier) vs. Regional/Ethnic dialects or children's speech (harder).

### Key Corpora
| Corpus | Description |
| :--- | :--- |
| **LibriSpeech** | 1000+ hours of read audiobooks (Clean vs. Other). |
| **Switchboard** | Telephone conversations between strangers (8 kHz). |
| **CHiME** | Robustness tasks (e.g., dinner parties with distant mics). |
| **Common Voice** | Crowdsourced multilingual read speech (33,000+ hours). |

---

## 2. Convolutional Neural Networks (CNNs) in ASR
CNNs are used as initial layers to extract features from audio (raw waveforms or Mel spectra).

### 1D Convolution for Speech
Unlike 2D CNNs for images, speech uses **1D convolution** by sliding a kernel over the time dimension.
1.  **Kernel (Filter):** A small vector of weights $w_1 \dots w_k$.
2.  **Convolution Operation ($*$):** Walking the kernel across the input and computing the dot product at each frame.
3.  **Padding ($p$):** Adding zeros to the start/end of the signal to keep the output length consistent.

### Mathematical Formula
For a kernel of width $k = 2p + 1$, the output $z_j$ at frame $j$ is:
$$z_j = \sum_{i=-p}^{p} x_{j+i} w_{i+p}$$

> **Numerical Example: 1D Convolution**
> *   **Input ($x$):** $[1, 2, 0, 3, 1]$
> *   **Kernel ($w$):** $[0.5, 1, 0.5]$ (Width $k=3$, so $p=1$)
> *   **Padding:** Add one zero at each end $\rightarrow [0, 1, 2, 0, 3, 1, 0]$
>
> **Calculation for $z_1$ (centered at $x_1=1$):**
> $z_1 = (0 \times 0.5) + (1 \times 1) + (2 \times 0.5) = 0 + 1 + 1 = \mathbf{2.0}$
>
> **Calculation for $z_2$ (centered at $x_2=2$):**
> $z_2 = (1 \times 0.5) + (2 \times 1) + (0 \times 0.5) = 0.5 + 2 + 0 = \mathbf{2.5}$

### Advanced CNN Concepts
*   **Stride:** The step size when moving the kernel. A stride of 2 halves the output length (subsampling).
*   **Depth:** If the input has 128 channels (e.g., Mel filters), the kernel must also have a depth of 128.
*   **Multiple Kernels:** To create an embedding of size $D$, we use $D$ separate kernels.

---

## 3. Encoder-Decoder Architecture (AED)
Commonly used in models like **OpenAI's Whisper**.

### Components
1.  **Encoder:** Processes acoustic features (e.g., Log-Mel spectrum) into hidden representations $H_{enc}$.
2.  **Decoder:** A conditional language model that generates text tokens one by one.
3.  **Cross-Attention:** The bridge where the decoder "looks" at the encoder's output.
    *   **Queries ($Q$):** From the previous decoder layer.
    *   **Keys ($K$) & Values ($V$):** From the encoder output.
    $$Q = H_{dec}W^Q, \quad K = H_{enc}W^K, \quad V = H_{enc}W^V$$
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Subsampling
Because speech is much longer than text (e.g., 25 frames per word), models use **subsampling** (via strided convolutions) to shorten the sequence before the encoder.

---

## 4. Self-Supervised Learning: HuBERT
**HuBERT (Hidden-unit BERT)** learns from unlabeled audio by predicting "hidden units."

### Two-Stage Training
1.  **Stage 1:** Cluster MFCC features using **K-means** to create 100 initial classes.
2.  **Stage 2:** Use the model's own intermediate representations to create 500 more refined classes.
3.  **Task:** Mask random segments of audio and train the model to predict the cluster ID of the masked frames.

### K-Means Clustering Algorithm
1.  **Assignment:** Assign each vector $v^{(i)}$ to the nearest centroid $\mu_j$.
    $$\text{cluster}(i) = \text{argmin}_j ||v^{(i)} - \mu_j||^2$$
2.  **Re-estimation:** Update centroids by taking the mean of all vectors assigned to them.
    $$\mu_i = \frac{1}{|S_i|} \sum_{v \in S_i} v$$

---

## 5. Connectionist Temporal Classification (CTC)
An alternative to Encoder-Decoder that maps input frames directly to output characters.

### Key Concepts
*   **Alignment:** A sequence of letters (one per frame) including a special **blank symbol ($\epsilon$)**.
*   **Collapsing Function ($B$):**
    1.  Merge consecutive identical characters (e.g., `aa` $\rightarrow$ `a`).
    2.  Remove all blanks (e.g., `a$\epsilon$b` $\rightarrow$ `ab`).
*   **Independence Assumption:** CTC assumes each frame's output is independent of others, so it needs an external **Language Model (LM)** for best results.

> **Numerical Example: CTC Collapsing**
> *   **Alignment:** `d d $\epsilon$ i i n $\epsilon$ n n e r r`
> *   **Step 1 (Merge):** `d $\epsilon$ i n $\epsilon$ n e r`
> *   **Step 2 (Remove $\epsilon$):** `dinner`

---

## 6. Evaluation: Word Error Rate (WER)
The standard metric for ASR performance.

### Formula
$$\text{WER} = 100 \times \frac{I + S + D}{N}$$
*   $I$: Insertions
*   $S$: Substitutions
*   $D$: Deletions
*   $N$: Total words in the reference transcript.

> **Numerical Example: WER Calculation**
> *   **REF:** "it is time for lunch" ($N=5$)
> *   **HYP:** "it was time lunch"
>
> **Analysis:**
> 1. "is" $\rightarrow$ "was" (**Substitution**)
> 2. "for" is missing (**Deletion**)
>
> **Counts:** $S=1, D=1, I=0$
> **WER:** $100 \times \frac{1 + 1 + 0}{5} = \mathbf{40\%}$

### Statistical Significance
*   **MAPSSWE:** The standard test to see if the difference between two systems' WER is statistically significant.
*   **McNemar's Test:** Not recommended for ASR because word errors are not independent.

---

## 7. Summary Table for Exam Prep
| Feature | Encoder-Decoder (AED) | CTC |
| :--- | :--- | :--- |
| **Alignment** | Handled by Attention | Handled by Blank Symbol & Collapsing |
| **LM** | Implicitly learned | Requires external LM |
| **Streaming** | Difficult (needs full context) | Easy (frame-by-frame) |
| **Accuracy** | Generally Higher | Generally Lower (unless using RNN-T) |
| **Loss** | Cross-Entropy | CTC Loss (Dynamic Programming) |
