---
theme: neversink
background: RL-bg.png
class: 'text-center'
# transition: slide-left
title: Speech Recognition (DSAI 456)
author: Mohamed Ghalwash
year: 2025-2026
venue: Zewail City
mdc: true
lecture: 2
---

# Speech Recognition <br> (DSAI 456)
## Lecture 2

Mohamed Ghalwash
<Email v="mghalwash@zewailcity.edu.eg" />

---
transition: fade-out
layout: top-title
---

:: title :: 

# Lecture 1 Recap 

:: content :: 

- Phonetics, Syllables 

- Signal Representation 
  
- Analog to digital 

- Transform the input waveform into a sequence of acoustic features 
  
- Features: pitch, loudness, intensity, F0 tracks


<BottomBar/>

---
layout: section
titlewidth: is-3
---


# What is 
  
- ### Frequency Spectrum

- ### Frequency Bands
  
- ### Frequency Spectrogram

- ### Mel Spectrum 

<BottomBar/>

---
layout: side-title
titlewidth: is-3
class: text-center
---
:: title :: 

# Frequency Spectrum 

:: content ::

<img src="./images/lec2-freq-sinewaves.png" />

<v-click>
<div class="text-red-500 text-xl flex items-center gap-2">
  <i class="i-mdi-emoticon-angry-outline"></i>
  <span>It does not look like a sine waveform</span>
</div>
</v-click>

<BottomBar/>

---
layout: side-title
titlewidth: is-3
---
:: title :: 

# Frequency Spectrum 

:: content ::

<img src="./images/lec2-freq-sinewaves2.png" />

 Maybe a combination of sine waveforms <span class="text-blue-500"> _with different frequencies and amplitude_</span>

<BottomBar/>

---
layout: side-title
titlewidth: is-3
---
:: title :: 

# Frequency Spectrum 

:: content ::

<img src="./images/lec2-freq-sinewaves3.png" />

<div class="flex items-center gap-2 text-yellow-400 text-4xl">
  <i class="i-ic-baseline-lightbulb-circle"></i>
  <span>Aha! I got it</span>
</div>

<BottomBar/>

---
layout: side-title
titlewidth: is-3
---
:: title :: 

# Frequency Spectrum 

:: content ::

<img src="./images/lec2-freq-sinewaves4.png" />

Done using <span class="text-blue-500"> Discrete Fourier Transform </span>
<BottomBar/>

---
layout: four-cell
---

:: top-left ::

<img src="./images/lec2-freq-sinewaves.png" />

:: bottom-left ::

<img src="./images/lec2-freq-sinewaves3.png" />

:: top-right ::

<img src="./images/lec2-freq-sinewaves2.png"  />

:: bottom-right ::

<img src="./images/lec2-freq-sinewaves4.png" />

<BottomBar/>

---
layout: side-title
titlewidth: is-3
---

:: title :: 

# Frequency Bands

:: content :: 

- Phones have characteristic spectral _signatures (Spectral peaks)_

- Inner ear computes the spectrum of the incoming waveform

- Spectrogram for three vowels
  
![](./images/lec2-freq-sinewaves6.png)

<!-- dark is more important because amplitude for white region is almost zero -->
<!-- which frequency range the dark area is -->

- Each dark bar is called a **formant** which is a frequency band that is particularly amplified by the vocal tract


<BottomBar/>


---
layout: full
titlewidth: is-3
---

![](./images/lec2-vowel-spectrum.png)




<BottomBar/>

---
layout: image-right
image: ./images/lec2_ear.png
---


# Issues with Hz

- Human hearing is less sensitive at higher frequencies
- The difference at low frequencies 50 Hz to 550 Hz is a massive change in perceived pitch
- However, the difference at high frequencies 13kHz to 15kHz is not a significant change in perceived pitch

- Information in low frequencies (like formants) is crucial for distinguishing vowels or nasals
- Modeling this human perceptual property improves speech recognition performance in the same way

<BottomBar/>
---
layout: top-title-two-cols
columns: is-6
align: l-lt-cm
---

:: title :: 

# Mel to the Rescue 

:: left :: 

- Designed to match human perception of pitch: how humans _perceive_ sound at different frequencies
- Equal intervals in mels represent equal perceived distances between pitches to a human listener
- The scale is anchored at 1000 Hz being equal to 1000 mels
- Below approximately 500 Hz, the mel and Hz scales are roughly equivalent

  $mel(f) = 2595 \log(1+\frac{f}{700})$

:: right :: 

![alt text](./images/lec2-mel-hz.png)

<BottomBar/>

---
layout: center
class: text-center
---

# Learn More

[Slidev](https://sli.dev) · [Course Homepage](https://github.com/m-fakhry/DSAI-456-SR)
