



```python
N = len(waveform)
```

👉 `N` = number of samples in your audio signal.

If your audio is 3 seconds long and your sample rate is 22050 Hz (samples per second),  
then:  

N = 3x 22050 = 66150  

So the waveform has 66,150 values (each one is a sound amplitude at a moment in time).

---

```python
X = np.fft.fft(waveform)
```

👉 This performs the **Fast Fourier Transform** (FFT).

- It converts your **waveform (time-based signal)** into its **frequency representation**.
    
- The result `X` is a complex array (each value has a real and imaginary part).
    
- Each element in `X` corresponds to a **specific frequency**, and the **complex value** encodes both:
    
    - amplitude (how strong that frequency is),
        
    - and phase (how it’s shifted in time).
        

So `X` tells you _which frequencies_ are present in your sound and _how strong_ they are.

---

```python
X_mag = np.abs(X)
```

👉 Takes the **magnitude** of the complex numbers from FFT.

- `np.abs()` gives the **amplitude** of each frequency component.
    
- This means: how strong each frequency is, ignoring phase.
    

Now `X_mag` is a real-valued array showing the **intensity** of each frequency.

---

```python
f = np.fft.fftfreq(N, d=1/sample_rate)
```

👉 This generates the **frequency values (in Hz)** that correspond to each index in the FFT.

- `N` → total number of samples
    
- `d = 1 / sample_rate` → the time between two samples
    
- The output `f` is an array that tells you:
    
    - what frequency (in Hz) each element of `X` represents
        

⚠️ Important:  
`np.fft.fftfreq()` returns both **positive and negative frequencies** (because the FFT of real signals is symmetric).

That’s why we usually only keep the positive half (see below).

---

```python
positive_frequencies = f[:N // 2]
positive_magnitudes = X_mag[:N // 2]
```

👉 Keep only the **first half** of the FFT result.

- For real signals, the FFT is **mirrored** around 0 Hz.  
    So the second half is just a mirror of the first.
    
- We take only the **first N/2 values** → the positive frequencies.
    

Now:

- `positive_frequencies` → from 0 Hz up to the Nyquist frequency (≈ sample_rate / 2)
    
- `positive_magnitudes` → amplitude for each of those frequencies
    

---

```python
plt.figure(figsize=(18, 4))
plt.plot(positive_frequencies, positive_magnitudes, color='blue')
plt.title('Frequency Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--')
plt.show()
```

👉 This part **plots the frequency spectrum** — the final visualization.

- **X-axis:** frequencies (in Hz)
    
- **Y-axis:** amplitude (strength of that frequency)
    
- The peaks in this graph show which frequencies are most dominant in your sound.
    

So if you had someone saying “Hi, how are you”, you might see peaks around:

- ~100–200 Hz (the voice’s pitch),
    
- and many more at higher frequencies (harmonics and consonant sounds).
    

---

### 🧠 Summary Table

|Step|Code|What it does|Result|
|---|---|---|---|
|1️⃣|`N = len(waveform)`|Count total samples|Number of data points|
|2️⃣|`np.fft.fft()`|Convert from time → frequency|Complex frequency spectrum|
|3️⃣|`np.abs()`|Get amplitude of each frequency|Strengths of all frequencies|
|4️⃣|`np.fft.fftfreq()`|Generate matching frequency values (Hz)|X-axis for plotting|
|5️⃣|`[:N // 2]`|Keep only positive frequencies|Avoid duplicates|
|6️⃣|`plt.plot()`|Visualize spectrum|Peaks = strong frequencies|

---
![[Pasted image 20251101184151.png]]
### 🎵 What the Plot Means

- **Left side (low frequencies):** bass, vowels, pitch
    
- **Right side (high frequencies):** sharp sounds, consonants, noise
    
- **Tall peaks:** dominant frequencies in the sound (like the speaker’s voice tone)
    
---

## 🎯 What This Part Does

This section:

1. Finds the **dominant frequencies** in your audio using FFT peaks.
    
2. Estimates the **true fundamental frequency (F₀)** using Librosa’s `pyin()` algorithm.
    
3. Draws a **spectrogram** with the pitch curve (F₀) over it — a very powerful way to visualize sound!
    

---

### 🌀 Step 1 — Find Dominant Frequencies using FFT Peaks

```python
peaks, _ = find_peaks(positive_magnitudes, height=np.max(positive_magnitudes)*0.1)
```

- `find_peaks()` comes from `scipy.signal`.
    
- It scans through the amplitude array (`positive_magnitudes`) and detects **local maxima (peaks)** — where amplitude goes up then down.
    
- `height=...` sets a **threshold** so that we ignore very small peaks (noise).  
    Here, only peaks **above 10%** of the maximum amplitude are kept.
    

So now, `peaks` contains the **indices** of strong frequencies.

---

```python
dominant_freqs = positive_frequencies[peaks[:5]]
print("Dominant Frequencies (Hz):", dominant_freqs)
```

- This picks the first 5 detected peaks (the strongest ones).
    
- Then it maps them back to their actual **frequency values in Hz**.
    
- Printing shows, for example:
    
    ```
    Dominant Frequencies (Hz): [139.7, 140.8, 142.0, 143.1, 144.3]
    ```
    
    These are the main frequency components present in your sound — they’re likely **harmonics** of the speaker’s voice.
    

---

### 🎨 Step 2 — Plot Frequency Spectrum with Peaks

```python
plt.figure(figsize=(18, 6))
plt.plot(positive_frequencies, positive_magnitudes, color='blue', label='Frequency Spectrum')
plt.plot(positive_frequencies[peaks], positive_magnitudes[peaks], 'ro', label='Peaks')
```

- Plots the full **frequency spectrum** (blue curve).
    
- Highlights the **peak frequencies** (red dots) on top.
    

---

```python
# Add text labels for the top 5 dominant frequencies
for i in range(min(5, len(peaks))):
    freq = positive_frequencies[peaks[i]]
    mag = positive_magnitudes[peaks[i]]
    plt.text(freq, mag, f"{freq:.1f} Hz", color='red', fontsize=10, ha='center', va='bottom')
```

- Loops through the top 5 peaks.
    
- Places text labels (`f"{freq:.1f} Hz"`) above each red dot — so you can see the actual frequency number on the graph.
    

---

```python
plt.title('Dominant Frequencies in Speech Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True, linestyle='--')
plt.show()
```

This produces a **frequency spectrum plot** showing:

- Blue curve = whole frequency content,
    
- Red dots = top frequencies,
    
- Red labels = frequency values in Hz.
    

✅ This graph visually identifies which frequencies dominate the sound (like the pitch and harmonics of the speaker’s voice).

---

## 🎵 Step 3 — Estimate Fundamental Frequency (F₀) using `pyin`

```python
f0, voiced_flag, voiced_prob = librosa.pyin(
    waveform,
    fmin=50,
    fmax=500,
    sr=sample_rate
)
```

- `librosa.pyin()` estimates the **fundamental frequency (F₀)** — the “main pitch” your ear hears.
    
- It works frame-by-frame, so each F₀ value corresponds to a small time segment.
    
- Parameters:
    
    - `fmin=50` and `fmax=500`: expected voice pitch range (suitable for speech).
        
    - `sr`: sampling rate.
        

**Outputs:**

- `f0`: estimated pitch per frame (Hz).  
    Some values can be `NaN` where the sound is **unvoiced** (e.g., silence or noisy parts).
    
- `voiced_flag`: boolean flag (True if frame has a clear pitch).
    
- `voiced_prob`: confidence of the estimate.
    

---

### 🧹 Clean F₀ Data

```python
times = librosa.times_like(f0)
f0_clean = np.where(np.isnan(f0), 0, f0)
```

- `librosa.times_like(f0)` → gives time (in seconds) for each F₀ value.
    
- `np.where(np.isnan(f0), 0, f0)` → replaces `NaN` (no pitch) with `0` to avoid breaks in the plot.
    

Now you have a clean F₀ array for plotting.

---

## 🌈 Step 4 — Compute and Plot the Spectrogram

A **spectrogram** shows how frequencies change over time.

```python
N_FFT = 2048
HOP_LENGTH = 512
D = librosa.stft(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
```

- `librosa.stft()` → computes the **Short-Time Fourier Transform**.
    
    - Splits the signal into small overlapping windows.
        
    - Computes FFT for each window (so you get frequency information over time).
        
- `N_FFT = 2048` → number of samples per window (controls frequency detail).
    
- `HOP_LENGTH = 512` → how much the window moves each time (controls time detail).
    
- `np.abs(D)` → converts complex numbers to amplitudes.
    
- `librosa.amplitude_to_db()` → converts amplitude to decibels for better visualization.
    

---

### 🎨 Step 5 — Plot Spectrogram with F₀ Overlay

```python
plt.figure(figsize=(15, 8))
librosa.display.specshow(S_db,
                         sr=sample_rate,
                         x_axis='time',
                         y_axis='hz',
                         hop_length=HOP_LENGTH,
                         cmap='magma')
```

- Displays the spectrogram as a heatmap:
    
    - **X-axis:** time
        
    - **Y-axis:** frequency (Hz)
        
    - **Color:** loudness (bright = strong, dark = weak)
        

---

```python
plt.colorbar(format='%+2.0f dB')
plt.title(f'Spectrogram (N_FFT={N_FFT}, HOP_LENGTH={HOP_LENGTH})')
```

Adds a colorbar (for dB scale) and a title.

---

```python
# Overlay the F0 curve
plt.plot(times, f0_clean, color='cyan', linewidth=2, label='F₀')
```

- Draws your estimated pitch (F₀) over the spectrogram.
    
- This helps visualize **how pitch changes over time** and how it matches the spectral energy bands.
    

---

```python
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()
```

Finishes the plot neatly with a legend and spacing.

---

## 🧠 Summary of What You’ve Done

|Step|Purpose|Output|
|---|---|---|
|`find_peaks()`|Detect strong frequencies in FFT|Top frequency components|
|`librosa.pyin()`|Estimate the fundamental pitch (F₀)|Real-time pitch curve|
|`librosa.stft()`|Compute frequency content across time|Spectrogram|
|`plt.plot(times, f0)`|Overlay pitch on spectrogram|Combined view of tone & harmonics|

---

### 🎯 Final Result

You get:

1. A **frequency spectrum** showing which frequencies are most dominant.
    
2. A **spectrogram** showing how frequency content evolves over time.
    
3. A **cyan line (F₀)** showing the pitch contour of your audio.
    

---

# GMM


```python
import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
  
```

```python


DATASET_PATH = "/root/.cache/kagglehub/datasets/pratt3000/vctk-corpus/versions/1/VCTK-Corpus/VCTK-Corpus/wav48"

N_MFCC = 13

K = 8

MAX_FILES = 5  # for demo

  

# 1) Extract MFCC


def get_mfcc(file):

    y, sr = librosa.load(file, sr=None)

    return librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).T

  

# ------------------------

# 2) Train GMM for each speaker

# ------------------------

def train_models():

    models = {}

    for sp in sorted(os.listdir(DATASET_PATH))[:5]:

        files = [os.path.join(DATASET_PATH, sp, f) for f in os.listdir(os.path.join(DATASET_PATH, sp)) if f.endswith(".wav")][:MAX_FILES]

        X = np.vstack([get_mfcc(f) for f in files])

        gmm = GaussianMixture(n_components=K, covariance_type='diag', max_iter=200, random_state=0)

        gmm.fit(X)

        models[sp] = gmm

    return models

  

# ------------------------

# 3) Predict speaker

# ------------------------

def predict(file, models):

    X = get_mfcc(file)

    scores = {sp: g.score(X) for sp, g in models.items()}

    return max(scores, key=scores.get)

  

# ------------------------

# 4) Evaluate accuracy

# ------------------------

def evaluate(models):

    y_true, y_pred = [], []

    for sp in sorted(os.listdir(DATASET_PATH))[:5]:

        files = [os.path.join(DATASET_PATH, sp, f) for f in os.listdir(os.path.join(DATASET_PATH, sp)) if f.endswith(".wav")][MAX_FILES:MAX_FILES+2]

        for f in files:

            y_true.append(sp)

            y_pred.append(predict(f, models))

    acc = accuracy_score(y_true, y_pred)

    print("Accuracy:", round(acc*100, 2), "%")

  

# ------------------------

# Run script

# ------------------------

models = train_models()

evaluate(models)
```