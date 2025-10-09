#  Speech Recognition (DSAI 456) - 2025-2026

Repository for the Speech Recognition undergraduate course (DSAI 456) for the 2025-2026 academic year at Zewail City University. 

---

### Logistics

Course | Speech Recognition - DSAI 456
---|----
Webpage| [https://github.com/m-fakhry/DSAI-456-Speech](https://github.com/m-fakhry/DSAI-456-Speech)
Structure | 2-hour lecture (Tue 8-10) and 3-hour lab (Sun 10-1, Sun 1-4, Mon 10-1, Tue 10-1, Tue 1-4)
TAs | Ahmed Aamer, Aya Nageh (Alphabetical order)
Book | "_Speech and Language Processing_", Jurafsky and Martin, 3rd Edition, 2025
Supplementary Book| "_Automatics Speech Recognition, A Deep Learning Approach_", Yu and Deng, 2015 
Objectives | Provide students with foundational knowledge and practical skills in the theory and application of speech processing and recognition. It covers both traditional statistical approaches and modern deep learning techniques to design, implement, and evaluate speech recognition systems.
Prerequitstis | Deep Learning, PyTorch, Python
Tools/APIs |  [librosa](https://librosa.org/doc/latest/index.html), [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis), [torchaudio](https://github.com/pytorch/audio), [openSmile](https://audeering.github.io/opensmile/), [senselab](https://github.com/sensein/senselab)
Tutorials | [Audio Signal Processing for ML, by Valerio Velardo](https://www.youtube.com/playlist?list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0)
Lab Policy| Assignments and quizzes

---

### Lectures

Week | Date |Topic | Contents | Lecture | Assignment
---|---|---|---|---|---
1| 09-23 | Intro  | What is speech recognition, applications, classical vs modern | | 
2| 09-30 | Foundations | phonetics and signals, frequency, amplitude, period, analog to digital, sampling and quantization, pitch, intensity | [Lecture 1](lectures/lec1.md) | [Assignment 1](assignments/assign1.md)
3| 10-07 | Acoustic Features | Discrete and fast Fourier transform,  time and frequency domain, freq spectrum, spectrogram, Mel scale | [Lecture 2](lectures/lec2.md) | [Assignment 2](assignments/assign2.md)
4| | Signal Processing | Mel filter bank, windowing, MFCC, Gaussian mixture models  | [Lecture XX] | [Assignment XX]
5| | Signal Processing | Hidden Markov models  | [Lecture XX] | [Assignment XX]
6| | Midterm | | | 
7| | CNN | Convolution neural network  for audio and ASR evaluation | [Lecture XX] | [Assignment XX]
8| | Enc-Dec | Encoder decoder architecture for audio, training, inference, and evaluation | [Lecture XX] | [Assignment XX]
9| | CTC | Connectionist temporal classification overview,  training and inference | [Lecture XX] | [Assignment XX]
10| | CTC | CTC hybrid with Enc-Dec and RNN| [Lecture XX] | [Assignment XX]
11| | TTS | Text to speech (TTS), use audio codec to learn tokens, ENCODEC model, vector quantization, residual vector quantization, generating audio with 2-stage language model (VALL-E), evaluation | [Lecture XX] | [Assignment XX]
12| | Paper Review | Students review recent papers about speech regonition and present to the entire class | | 
13| | Paper Review | Students review recent papers about speech regonition and present to the entire class | | 
14| | Final | | | 

Please note that the syllabus content is subject to change throughout the semester. Topics may be added or removed based on the instructor’s discretion, student progress, and available time. Your feedback and participation will inform these adjustments to ensure alignment with course goals and schedule constraints.

--- 

### Grading Policy 

Topic| Percentage | Notes
---|---|---
Lab Assignments | 10% | 
Lab Quizzes | 10% | 
Final Lab | 10% | 
Paper Presentation | 10% | 
Midterm | 20% | 
Final | 40% | 
