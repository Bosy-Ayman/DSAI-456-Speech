

This question bank is based on the provided documents, covering core concepts in speech feature extraction, density estimation, and sequence modeling.

## Part 1: Understanding Questions

These questions test your conceptual understanding of the key models and processes.

**Q1: What is the primary purpose of computing Mel Frequency Cepstral Coefficaients (MFCCs) for speech recognition?**

**A:** The primary purpose is to create a compact, robust feature vector that represents the phonetically relevant information from a frame of speech. It achieves this by:

1. Modeling the human auditory system's non-linear frequency perception using the **mel scale**. This scale emphasizes lower frequencies, where most phonetically important information (like formants) resides, and compresses higher frequencies.
    
2. Separating the vocal tract filter characteristics (which determine the phone being spoken) from the glottal source (fundamental frequency). This separation is accomplished by taking the **cepstrum** (via the Discrete Cosine Transform), which effectively isolates the slow-changing vocal tract shape (in the low-order coefficients) from the fast-changing source pitch.
    

**Q1a: What are "delta" and "double-delta" coefficients in the context of MFCCs, and why are they important?**

**A:**

- **Delta coefficients** (or "velocity" features) are the first-order derivatives of the MFCC coefficients, calculated over a small time window. They represent the _rate of change_ of the spectral features.
    
- **Double-delta coefficients** (or "acceleration" features) are the second-order derivatives, representing the _rate of change of the deltas_.
    
- **Importance:** Speech is a _dynamic_ signal, not a series of static snapshots. The way the spectrum _changes_ (e.g., the transition from a consonant to a vowel) is often more informative for recognition than the static sound itself. Deltas and double-deltas capture this crucial temporal (dynamic) information, providing the model with context about the recent past and future of the signal.
    

**Q2: Why is a single Gaussian distribution often insufficient for modeling speech features, leading to the use of Gaussian Mixture Models (GMMs)?**

**A:** A single Gaussian distribution is ==unimodal (it has only one peak)== and can only model a simple, convex-shaped cluster of data. Speech feature data for a given sound (e.g., a single HMM state) is often far more complex and **multimodal** (having multiple peaks). This complexity arises from:

- **Speaker variations:** Different speakers (e.g., with different vocal tract sizes, accents, or genders) produce acoustically different versions of the same phoneme.
    
- **Coarticulation:** The sound of a phoneme is influenced by its neighbors (e.g., the 'k' sound in "key" is different from the 'k' in "coo").
    
- **Acoustic environment:** Background noise or reverberation can distort the features in various ways. A GMM uses a flexible, weighted sum of multiple Gaussian components to accurately model these complex, real-world distributions.
    

**for HMMs, as defined in the Rabiner tutorial, and what algorithm is used to solve each?**

**A:**

1. **Problem 1 (Evaluation):** Given an observation sequence $O$ and a model $\lambda$, what is the probability of the observation sequenc totale given the model, $P(O|\lambda)$?
    
    - **Solution:** The **Forward Algorithm**.
        
    
    This is used to "score" how well a model matches an observation.
    
2. **Problem 2 (Decoding):** Given an observation sequence $O$ and a model $\lambda$, what is the most likely _sequence of hidden states_ $Q$ that produced the observations?
    
    - **Solution:** The **Viterbi Algorithm**.
        
        This is used for alignment and segmentation.
        
3. **Problem 3 (Training):** Given an observation sequence $O$, how d one or more observation sequences$\lambda = (A, B, \pi)$ to maximize the probability $P(O|\lambda)$?
    
    - **Solution:** The **Baum-Welch Algorithm** (a specific instance of the Expectation-Maximization (EM) algorithm).
        

**Q3a: What do the $\pi$,** $A$**, and** $B$ **parameters of an HMM represent?**

**A:**

- $\pi$ **(Pi):** The **Initial State Distribution**. This is a vector of probabilities that defines the chance of starting in any given state. For example, in a 5-state model, $\pi_1=1.0, \pi_{2...5}=0.0$ would mean every sequence _must_ start in state 1.
    
- $A$**:** The **State Transition Matrix**. This matrix defines the probability of moving from one state to another. The entry $a_{ij}$ is the probability of moving from state $i$ to state $j$ at the next time step.
    
- $B$**:** The **Observation Probability Distribution**. This defines the probability of emitting (or "observing") a particular feature vector (e.g., an MFCC vector) _given_ that the model is in a certain state. In an HMM-GMM, the $B$ parameter for state $j$ is a full GMM, $b_j(O)$.
    

**Q4: What is the "responsibility" of a mixture component in a GMM, and in which step of the EM algorithm is it calculated?**

**A:** The "responsibility" (denoted $r_{nk}$ or $\gamma_t(k)$ in the texts) is the posterior probability that a specific data point $x_n$ was generated by a specific mixture component $k$$k$. It's a "soft assignment" of the data point to that component, calculated based on the data point's proximity to all components. For example, a point might be 90% "responsible" to component 1 and 10% to component 2.**step (Expectation step)** of the EM algorithm, based on the current model parameters.

**Q5: What is the purpose of "windowing" (e.g., with a Hamming window) in speech feature extraction?**

**A:** Speech is a non-stationary signal (its statistical properties change over time). To perform spectral analysis (like a DFT), we need to analyze short, quasi-stationary segments. This is done by applying a "window" (e.g., 25ms wide). A rectangular window causes abrupt cutoffs, creating spectral artifacts. A Hamming window tapers the signal to zero at the window's edges, minimizing these discontinuities and resulting in a cleaner, more accurate spectrum.

## Part 2: Comparison Questions

These questions ask you to compare and contrast key concepts.

**Q1: Compare and contrast a Gaussian Mixture Model (GMM) and a Hidden Markov Model (HMM). What is their typical relationship in a speech recognition system?**

**A:**

- **GMM:** A GMM is a static model for **density estimation**. It models the probability distribution of a _single_ data vector (e.g., an MFCC vector) and has no concept of time or sequence.
    
- **HMM:** An HMM is a dynamic model for **sequences**. It models the temporal probability of a _sequence_ of observations, using hidden states and transitions to capture how the signal changes over time.
    
- **Relationship:** In a classic HMM-GMM speech recognition system, the HMM models the temporal structure of speech (e.g., how a word's phonemes follow one another). Each **state** of the HMM then uses a **GMM** as its output probability distribution, modeling the probability of observing a specific MFCC vector _while in that state_.
    

**Q2: What is the main difference between Log Mel Spectrum features and MFCC features? What additional step is required to get from the former to the latter?**

**A:**

- **Log Mel Spectrum:** This is a vector of log-energies from a mel-spaced filter bank. It represents the _spectrum_ of the signal, weighted according to human perception. Its components are correlated.
    
- **MFCC (Mel Frequency Cepstral Coefficients):** This is a vector of coefficients derived _from_ the log mel spectrum. It represents the _cepstrum_ (the spectrum of the spectrum).
    
- **Additional Step:** To get from the log mel spectrum to MFCCs, you must apply the **Discrete Cosine Transform (DCT)**. This step de-correlates the features and effectively separates the vocal tract filter information (captured in the low-order MFCCs) from the source excitation.
    

**Q3: Compare an ergodic HMM with a left-right HMM. Which one is more suitable for modeling speech, and why?**

**A:**

- **Ergodic HMM:** A fully connected HMM where a transition is possible from any state to any other state.
    
- **Left-Right HMM (Bakis Model):** A constrained HMM where transitions are only allowed to the same state or to states with a higher index (i.e., $a_{ij} = 0 \text{ for } j < i$). The state index must be non-decreasing.
    
- **Suitability:** The **left-right HMM** is far more suitable for modeling signals like speech, which are inherently progressive. A word, for example, moves from a beginning sound to a middle sound to an end sound. It does not go from the end sound back to the beginning. The left-right topology enforces this forward-moving temporal structure.
    

**Q4: Compare the Viterbi algorithm and the Forward algorithm. What does each one compute, and how does their core mathematical operation differ?**

**A:** Both are dynamic programming algorithms that process an observation sequence $O$ given an HMM $\lambda$.

- **Forward Algorithm:** Computes the **total probability** $P(O|\lambda)$ by summing the probabilities of _all possible_ state paths that could have generated $O$. Its core operation at each step is a **summation** over the probabilities of previous states.
    
- **Viterbi Algorithm:** Computes the probability of the **single best path** (the most likely state sequence) and identifies that path. Its core operation at each step is a **maximization** (finding the `max`) over the probabilities of previous states.
    

## Part 3: Mathematical Problem Questions

These questions require you to apply the formulas from the texts.

**Q1: (GMM Responsibility)** You have a 1D data point $x_n = 4$ and a GMM with two components ($K=2$):

- Component 1: $\pi_1 = 0.5$, $\mu_1 = 2$, $\sigma_1^2 = 2$
    
- Component 2: $\pi_2 = 0.5$, $\mu_2 = 7$, $\sigma_2^2 = 1$
    
- The Gaussian PDF is: $N(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
    

Calculate the **responsibility (**$r_{n1}$**)** of the _first_ component for this data point. (You can leave $\pi$ and $e$ in your final answer).

**A:**

1. **Calculate likelihoods** for $x_n = 4$:
    
    - $N(4|C_1) = \frac{1}{\sqrt{2\pi \cdot 2}} \exp\left(-\frac{(4-2)^2}{2 \cdot 2}\right) = \frac{1}{\sqrt{4\pi}} \exp\left(-\frac{4}{4}\right) = \frac{1}{2\sqrt{\pi}} e^{-1}$
        
    - $N(4|C_2) = \frac{1}{\sqrt{2\pi \cdot 1}} \exp\left(-\frac{(4-7)^2}{2 \cdot 1}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{9}{2}\right) = \frac{1}{\sqrt{2\pi}} e^{-4.5}$
        
2. **Calculate responsibility** $r_{n1} = \frac{\pi_1 N(4|C_1)}{\pi_1 N(4|C_1) + \pi_2 N(4|C_2)}$:
    
    - $r_{n1} = \frac{0.5 \cdot \frac{1}{2\sqrt{\pi}} e^{-1}}{ (0.5 \cdot \frac{1}{2\sqrt{\pi}} e^{-1}) + (0.5 \cdot \frac{1}{\sqrt{2\pi}} e^{-4.5}) }$
        
    - _Simplify (divide all terms by_ $0.5 / \sqrt{2\pi}$_):_
        
    - $r_{n1} = \frac{ \frac{1}{\sqrt{2}} e^{-1} }{ (\frac{1}{\sqrt{2}} e^{-1}) + (e^{-4.5}) } = \frac{e^{-1}/\sqrt{2}}{e^{-1}/\sqrt{2} + e^{-4.5}}$
        
    
    _(Approximate value:_ $e^{-1} \approx 0.368$_,_ $e^{-4.5} \approx 0.011$_._ $r_{n1} \approx \frac{0.368 / 1.414}{(0.368 / 1.414) + 0.011} = \frac{0.260}{0.260 + 0.011} = \frac{0.260}{0.271} \approx 0.959$_)_
    

**Q2: (GMM M-Step)** In the M-step of the EM algorithm for a GMM, you are updating the mean for the $k$-th component, $\mu_k$. You have $N=3$ data points ($x_1=10, x_2=20, x_3=30$) and the responsibilities from the E-step for component $k=1$ are:

- $r_{11}=0.8$ (for $x_1$)
    
- $r_{21}=0.5$ (for $x_2$)
    
- $r_{31}=0.1$ (for $x_3$) What is the new mean $\mu_1^{\text{new}}$?
    

**A:** The formula for the new mean $\mu_k^{\text{new}}$ is the responsibility-weighted average of the data: $\mu_k^{\text{new}} = \frac{\sum_{n=1}^N r_{nk} x_n}{\sum_{n=1}^N r_{nk}}$

1. **Numerator (weighted sum of data):**
    
    - $\sum_{n=1}^3 r_{n1} x_n = (r_{11} \cdot x_1) + (r_{21} \cdot x_2) + (r_{31} \cdot x_3)$
        
    - $= (0.8 \cdot 10) + (0.5 \cdot 20) + (0.1 \cdot 30)$
        
    - $= 8 + 10 + 3 = 21$
        
2. **Denominator (total responsibility for component 1,** $N_1$**):**
    
    - $N_1 = \sum_{n=1}^3 r_{n1} = 0.8 + 0.5 + 0.1 = 1.4$
        
3. **New Mean:**
    
    - $\mu_1^{\text{new}} = \frac{21}{1.4} = 15$
        

**Q3: (Mel Scale)** The formula to convert frequency $f$ in Hz to mels is: $mel(f) = 1127 \ln(1 + f/700)$. What is the perceived pitch in mels of a 700 Hz tone?

**A:** We plug $f=700$ into the formula:

- $mel(700) = 1127 \ln(1 + 700/700)$
    
- $mel(700) = 1127 \ln(1 + 1)$
    
- $mel(700) = 1127 \ln(2)$
    
- _(Approx._ $1127 \cdot 0.693 \approx 781$ _mels)_
    

**Answer:** $1127 \ln(2)$ mels.