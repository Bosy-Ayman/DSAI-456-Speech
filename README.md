#  Speech Recognition (DSAI 456) - 2025-2026

Repository for the Speech Recognition undergraduate course (DSAI 456) for the 2025-2026 academic year at Zewail City University. 

---

### Logistics

Course | Speech Recognition - DSAI 456
---|----
Webpage| [https://github.com/m-fakhry/DSAI-456-Speech](https://github.com/m-fakhry/DSAI-456-Speech)
Structure | 2-hour lecture (Tue 8-10) and 3-hour lab (Sun 10-1, Sun 1-4, Mon 10-1, Tue 10-1, Tue 1-4)
TAs | Ahmed Aamer, Aya Nageh, Ossam Ghandour (Alphabetical order)
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
4| 10-14 | Acoustic Features | Mel filter bank, windowing, MFCC | [Lecture 3](lectures/lec3.md)  | [Assignment 3](assignments/assign3.md)
5| 10-21 | GMM | Gaussian mixture models  | [Lecture 4](lectures/lec4.md) | [Assignment 4](assignments/assign4.md)
6| 10-28 | HMM | Hidden Markov models, evaluation problem | [Lecture 5](lectures/lec5.md) | No assignment as preparation for the quiz
7| 11-4 | HMM | Evaluation problem, direct computation and forward algorithm | [Lecture 6](lectures/lec6.md) | No assignment as preparation for the midterm
8| 11-11 | Midterm | | | 
9| 11-18 | HMM | Decoding problem, direct and Viterbi algorithm | [Lecture 7](lectures/lec7.md) | [Assignment 5](assignments/assign5.md)
10| 11-25 | CTC | Connectionist temporal classification overview,  training and inference |  Delivered by Eng. Ossama Ghandour | [Assignment 6](assignments/assign6.md)
11| 12-2 | CNN, Enc-Dec | Convolution neural network  for audio and ASR evaluation, Encoder decoder architecture for audio, training, inference, and evaluation, CTC hybrid with Enc-Dec and RNN | [Lecture XX] | -
12| 12-9 | TTS | LAB QUIZ, Text to speech (TTS), use audio codec to learn tokens, ENCODEC model, vector quantization, residual vector quantization | [Lecture 10](lectures/lec10.md) | -
13| 12-16 | TTS | FINAL LAB, generating audio with 2-stage language model (VALL-E) | [Lecture XX] | -
14| 12-23 | Paper Review | Students review recent papers about speech recognition and present to the entire class | | 
15| 12-30 | Paper Review | Students review recent papers about speech recognition and present to the entire class | | 
16| 1-7 | Final | | | 

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
