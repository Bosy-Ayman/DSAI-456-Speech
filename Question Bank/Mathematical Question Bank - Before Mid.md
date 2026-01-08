This guide provides a comprehensive review of potential mathematical problems for the midterm, based on the course material. It covers Mel-Frequency Conversion, Gaussian Mixture Models (GMMs), and Hidden Markov Models (HMMs).

---

## 1. Mel-Frequency Conversion

### Concept

The **Mel scale** is a perceptual scale of pitches. It is designed so that pitches judged by listeners to be equal in distance from one another are also equidistant on the scale. This scale reflects human hearing more accurately than the linear Hertz (Hz) scale.

### Formula

The conversion from frequency $f$ (in Hz) to Mels is given by:

$$mel(f) = 1127 \cdot \ln\left(1 + \frac{f}{700}\right)$$

### Worked Example: Convert 1000 Hz

**Problem:** Convert a frequency of 1000 Hz to the Mel scale.

**Solution:**

1. Plug $f = 1000$ into the formula:
    
    $mel(1000) = 1127 \cdot \ln\left(1 + \frac{1000}{700}\right)$
    
2. Calculate the fraction:
    
    $mel(1000) = 1127 \cdot \ln(1 + 1.4286) = 1127 \cdot \ln(2.4286)$
    
3. Calculate the natural log:
    
    $\ln(2.4286) \approx 0.8872$
    
4. Multiply to get the final answer:
    
    $mel(1000) = 1127 \cdot 0.8872 \approx 1000 \text{ Mels}$
    
    (By design, 1000 Hz is approximately 1000 Mels.)
    

### Practice Problem: Convert 4500 Hz

**Problem:** Your feature extraction process identifies a strong formant at 4500 Hz. Convert this frequency to the Mel scale.

Solution:

$mel(4500) = 1127 \cdot \ln\left(1 + \frac{4500}{700}\right)$

$mel(4500) = 1127 \cdot \ln(1 + 6.4286) = 1127 \cdot \ln(7.4286)$

$mel(4500) = 1127 \cdot 2.0053 \approx 2260 \text{ Mels}$

---

## 2. Gaussian Mixture Models (GMMs)

### A. MLE for a Single Gaussian Distribution

Concept:

Maximum Likelihood Estimation (MLE) for a single Gaussian distribution finds the parameters (mean $\mu$ and variance $\sigma^2$) that best fit a given dataset. For a single Gaussian, the MLE solution is simply the sample mean and sample variance.

**Formulas:**

- Sample Mean ($\hat{\mu}$):
    
    $$\hat{\mu} = \frac{1}{N} \sum_{i=1}^{N} x_i$$
    
- Sample Variance ($\hat{\sigma}^2$):
    
    $$\hat{\sigma}^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \hat{\mu})^2$$
    

Worked Example:

Problem: Given the dataset $X = [2, 4, 9]$, find the MLE estimates for the mean ($\hat{\mu}$) and variance ($\hat{\sigma}^2$).

**Solution:**

1. Calculate the Mean:
    
    $\hat{\mu} = \frac{2 + 4 + 9}{3} = \frac{15}{3} = 5$
    
2. Calculate the Variance:
    
    $\hat{\sigma}^2 = \frac{(2 - 5)^2 + (4 - 5)^2 + (9 - 5)^2}{3}$
    
    $\hat{\sigma}^2 = \frac{(-3)^2 + (-1)^2 + (4)^2}{3} = \frac{9 + 1 + 16}{3} = \frac{26}{3} \approx 8.67$
    

### B. Calculating GMM Probability Density $p(x)$

Concept:

A GMM's probability density is the weighted sum of the densities of its individual Gaussian components.

**Formulas:**

- 1D Gaussian Density $N(x | \mu, \sigma^2)$:
    
    $$N(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
    
- GMM Density $p(x)$:
    
    $$p(x) = \sum_{k=1}^{K} \pi_k N(x | \mu_k, \sigma_k^2)$$
    
    (Where $\pi_k$ is the weight of component $k$, and $\sum \pi_k = 1$)
    

**Worked Example: Calculate $p(x=3)$**

Problem: Given a 1D GMM with 2 components, calculate the probability density $p(x)$ for a data point $x = 3$.

- **Component 1:** $\pi_1 = 0.4$, $\mu_1 = 2$, $\sigma_1^2 = 1$
    
- **Component 2:** $\pi_2 = 0.6$, $\mu_2 = 5$, $\sigma_2^2 = 2$
    

**Solution:**

1. Calculate $N(3 | \mu_1, \sigma_1^2)$: (for $\mu_1=2, \sigma_1^2=1$)
    
    $N(3 | 2, 1) = \frac{1}{\sqrt{2\pi \cdot 1}} \exp\left(-\frac{(3 - 2)^2}{2 \cdot 1}\right) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{1}{2}\right)$
    
    $\approx (0.399) \cdot (0.606) \approx 0.242$
    
2. Calculate $N(3 | \mu_2, \sigma_2^2)$: (for $\mu_2=5, \sigma_2^2=2$)
    
    $N(3 | 5, 2) = \frac{1}{\sqrt{2\pi \cdot 2}} \exp\left(-\frac{(3 - 5)^2}{2 \cdot 2}\right) = \frac{1}{\sqrt{4\pi}} \exp\left(-\frac{4}{4}\right)$
    
    $\approx (0.282) \cdot \exp(-1) \approx (0.282) \cdot (0.368) \approx 0.104$
    
3. Combine using weights:
    
    $p(3) = \pi_1 N(3 | ...) + \pi_2 N(3 | ...)$
    
    $p(3) = (0.4 \cdot 0.242) + (0.6 \cdot 0.104)$
    
    $p(3) = 0.0968 + 0.0624 = 0.1592$
    

### C. Calculating Responsibilities (E-Step)

Concept:

The "responsibility" $r_{nk}$ is the posterior probability that component $k$ was responsible for generating a data point $x_n$.

Formula:

$$r_{nk} = \gamma(z_{nk}) = \frac{\pi_k N(x_n | \mu_k, \sigma_k^2)}{\sum_{j=1}^{K} \pi_j N(x_n | \mu_j, \sigma_j^2)} = \frac{\text{Weighted density of component } k}{\text{Total density } p(x_n)}$$

Worked Example:

Problem: Using the same GMM and $p(3) = 0.1592$ from the previous example, calculate the responsibilities of Component 1 and Component 2 for the data point $x = 3$.

**Solution:**

1. Responsibility of Component 1 ($r_{3,1}$):
    
    $r_{3,1} = \frac{\pi_1 N(3 | ...)}{p(3)} = \frac{0.4 \cdot 0.242}{0.1592} = \frac{0.0968}{0.1592} \approx 0.608$
    
2. Responsibility of Component 2 ($r_{3,2}$):
    
    $r_{3,2} = \frac{\pi_2 N(3 | ...)}{p(3)} = \frac{0.6 \cdot 0.104}{0.1592} = \frac{0.0624}{0.1592} \approx 0.392$
    
    (Note: Responsibilities for a data point sum to 1. $0.608 + 0.392 = 1.0$)
    

### D. Practice Problem: GMM Calculations for $x=1$

**Problem:** Using the same 2-component GMM as before:

- **Component 1:** $\pi_1 = 0.4$, $\mu_1 = 2$, $\sigma_1^2 = 1$
    
- Component 2: $\pi_2 = 0.6$, $\mu_2 = 5$, $\sigma_2^2 = 2$
    
    Calculate:
    

1. The total probability density $p(x)$ for a data point $x = 1$.
    
2. The responsibilities of Component 1 and Component 2 for $x = 1$.
    

**Solution:**

1. **Calculate Gaussian Densities for $x=1$:**
    
    - $N(1 | 2, 1) = \frac{1}{\sqrt{2\pi \cdot 1}} \exp\left(-\frac{(1 - 2)^2}{2 \cdot 1}\right) = 0.399 \cdot \exp(-0.5) \approx 0.242$
        
    - $N(1 | 5, 2) = \frac{1}{\sqrt{2\pi \cdot 2}} \exp\left(-\frac{(1 - 5)^2}{2 \cdot 2}\right) = 0.282 \cdot \exp\left(-\frac{16}{4}\right) = 0.282 \cdot \exp(-4) \approx 0.005$
        
2. **Calculate Total Probability Density $p(1)$:**
    
    - $p(1) = (0.4 \cdot 0.242) + (0.6 \cdot 0.005)$
        
    - $p(1) = 0.0968 + 0.003 = 0.0998$
        
3. **Calculate Responsibilities:**
    
    - $r_{1,1} = \frac{0.0968}{0.0998} \approx 0.97$
        
    - $r_{1,2} = \frac{0.003}{0.0998} \approx 0.03$
        

### E. GMM Parameter Re-estimation (MLE via EM)

Concept:

Given a dataset and an initial GMM, we perform one iteration of the Expectation-Maximization (EM) algorithm to find a new set of parameters ($\pi_k, \mu_k, \sigma_k^2$) that better fits the data (i.e., increases the likelihood).

Formulas (M-Step for 1D GMMs):

First, calculate the "soft count" (total responsibility) for each component $k$:

$$N_k = \sum_{n=1}^{N} r_{nk}$$

Then, update the parameters:

1. **New Weight:** $\pi_k^{new} = \frac{N_k}{N}$ (where $N$ is the total number of data points)
    
2. **New Mean:** $\mu_k^{new} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} x_n$
    
3. **New Variance:** $(\sigma_k^2)^{new} = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} (x_n - \mu_k^{new})^2$
    

Worked Example: One EM Iteration

Problem: Given the following GMM and data, compute the updated parameters after one iteration.

- **Initial GMM:**
    
    - C1: $\pi_1 = 0.5$, $\mu_1 = 1$, $\sigma_1^2 = 1$
        
    - C2: $\pi_2 = 0.5$, $\mu_2 = 5$, $\sigma_2^2 = 1$
        
- **Data ($N=2$):** $x_1 = 2$, $x_2 = 4$
    

**Solution:**

==**1. E-Step: Calculate Responsibilities**==

- **For data point $x_1 = 2$:**
    
    - $N(2 | 1, 1) = 0.399 \cdot \exp\left(-\frac{(2 - 1)^2}{2 \cdot 1}\right) = 0.399 \cdot \exp(-0.5) \approx 0.242$
        
    - $N(2 | 5, 1) = 0.399 \cdot \exp\left(-\frac{(2 - 5)^2}{2 \cdot 1}\right) = 0.399 \cdot \exp(-4.5) \approx 0.004$
        
    - $p(2) = 0.5(0.242) + 0.5(0.004) = 0.121 + 0.002 = 0.123$
        
    - **Responsibilities for $x_1=2$:**
        
        - $r_{1,1} = 0.121 / 0.123 \approx 0.98$
            
        - $r_{1,2} = 0.002 / 0.123 \approx 0.02$
            
- **For data point $x_2 = 4$:**
    
    - $N(4 | 1, 1) = 0.399 \cdot \exp\left(-\frac{(4 - 1)^2}{2 \cdot 1}\right) = 0.399 \cdot \exp(-4.5) \approx 0.004$
        
    - $N(4 | 5, 1) = 0.399 \cdot \exp\left(-\frac{(4 - 5)^2}{2 \cdot 1}\right) = 0.399 \cdot \exp(-0.5) \approx 0.242$
        
    - $p(4) = 0.5(0.004) + 0.5(0.242) = 0.002 + 0.121 = 0.123$
        
    - **Responsibilities for $x_2=4$:**
        
        - $r_{2,1} = 0.002 / 0.123 \approx 0.02$
            
        - $r_{2,2} = 0.121 / 0.123 \approx 0.98$
            

==**2. M-Step: Re-estimate Parameters**==

- **Responsibility Matrix:**
    
    - $r_{1,1} = 0.98$, $r_{1,2} = 0.02$
        
    - $r_{2,1} = 0.02$, $r_{2,2} = 0.98$
        
- **Calculate Soft Counts ($N_k$):**
    
    - $N_1 = r_{1,1} + r_{2,1} = 0.98 + 0.02 = 1.0$
        
    - $N_2 = r_{1,2} + r_{2,2} = 0.02 + 0.98 = 1.0$
        
- **Update Weights ($\pi_k^{new}$):** (Total data $N=2$)
    
    - $\pi_1^{new} = N_1 / N = 1.0 / 2 = 0.5$
        
    - $\pi_2^{new} = N_2 / N = 1.0 / 2 = 0.5$
        
- **Update Means ($\mu_k^{new}$):**
    
    - $\mu_1^{new} = \frac{r_{1,1}x_1 + r_{2,1}x_2}{N_1} = \frac{0.98(2) + 0.02(4)}{1.0} = 1.96 + 0.08 = 2.04$
        
    - $\mu_2^{new} = \frac{r_{1,2}x_1 + r_{2,2}x_2}{N_2} = \frac{0.02(2) + 0.98(4)}{1.0} = 0.04 + 3.92 = 3.96$
        
- **Update Variances ($(\sigma_k^2)^{new}$):**
    
    - $(\sigma_1^2)^{new} = \frac{r_{1,1}(x_1 - \mu_1^{new})^2 + r_{2,1}(x_2 - \mu_1^{new})^2}{N_1}$
        
    - $= \frac{0.98(2 - 2.04)^2 + 0.02(4 - 2.04)^2}{1.0}$
        
    - $= 0.98(-0.04)^2 + 0.02(1.96)^2 = 0.98(0.0016) + 0.02(3.8416) \approx 0.00157 + 0.0768 \approx 0.078$
        
    - $(\sigma_2^2)^{new} = \frac{r_{1,2}(x_1 - \mu_2^{new})^2 + r_{2,2}(x_2 - \mu_2^{new})^2}{N_2}$
        
    - $= \frac{0.02(2 - 3.96)^2 + 0.98(4 - 3.96)^2}{1.0}$
        
    - $= 0.02(-1.96)^2 + 0.98(0.04)^2 = 0.02(3.8416) + 0.98(0.0016) \approx 0.0768 + 0.00157 \approx 0.078$
        

**Summary of Results:**

- **Initial Parameters:** $\pi = [0.5, 0.5]$, $\mu = [1, 5]$, $\sigma^2 = [1, 1]$
    
- **New Parameters:** $\pi^{new} = [0.5, 0.5]$, $\mu^{new} = [2.04, 3.96]$, $(\sigma^2)^{new} = [0.078, 0.078]$
    

---

## 3. Hidden Markov Models (HMMs)

For all HMM examples and problems below, we will use the following model ($\lambda$):

- **States:** S1, S2
    
- **Initial Probabilities ($\pi$):** $\pi_1 = 0.8$, $\pi_2 = 0.2$
    
- **Transition Matrix (A):**
    
    - $a_{11}$ (S1 $\to$ S1) = 0.6
        
    - $a_{12}$ (S1 $\to$ S2) = 0.4
        
    - $a_{21}$ (S2 $\to$ S1) = 0.3
        
    - $a_{22}$ (S2 $\to$ S2) = 0.7
        
- **Emission Matrix (B):**
    
    - $b_1(A) = 0.5$, $b_1(B) = 0.5$
        
    - $b_2(A) = 0.2$, $b_2(B) = 0.8$
        

### A. Path Probability (Direct Computation)

Concept:

Calculating the joint probability of a specific observation sequence $O$ AND a specific state sequence $Q$, given the model $\lambda$.

Formula:

$$P(O,Q | \lambda) = \pi_{q_1} b_{q_1}(O_1) \prod_{t=2}^{T} a_{q_{t-1}, q_t} b_{q_t}(O_t)$$

(Initial prob $\cdot$ First emission $\cdot$ [Product of (transition $\cdot$ emission) for all following steps])

Worked Example:

Problem: Calculate $P(O,Q | \lambda)$ for:

- Observation Sequence $O = (A, B)$
    
- State Sequence $Q = (S1, S2)$
    

Solution:

$P = \pi_{S1} \cdot b_{S1}(A) \cdot a_{S1,S2} \cdot b_{S2}(B)$

$P = (0.8) \cdot (0.5) \cdot (0.4) \cdot (0.8)$

$P = 0.128$

### Practice Problem: Path Probability

**Problem:** Using the same HMM, calculate $P(O,Q | \lambda)$ for:

- Observation Sequence $O = (A, A)$
    
- State Sequence $Q = (S1, S1)$
    

Solution:

$P = \pi_{S1} \cdot b_{S1}(A) \cdot a_{S1,S1} \cdot b_{S1}(A)$

$P = (0.8) \cdot (0.5) \cdot (0.6) \cdot (0.5)$

$P = 0.120$

### B. HMM Evaluation: The Forward Algorithm

Concept:

Efficiently computes the total probability of an observation sequence $O$, $P(O | \lambda)$, by summing the probabilities of all possible state paths that could have generated it.

Core Variable:

$\alpha_t(i) = P(O_1, ..., O_t, q_t = i | \lambda)$

This is the probability of the partial observation sequence up to time $t$, ending in state $i$.

**Formulas:**

1. Initialization ($t=1$):
    
    $\alpha_1(i) = \pi_i b_i(O_1)$
    
2. Induction (for $t=1$ to $T-1$):
    
    $\alpha_{t+1}(j) = \left[ \sum_{i=1}^{N} \alpha_t(i) a_{ij} \right] b_j(O_{t+1})$
    
3. Termination:
    
    $P(O | \lambda) = \sum_{i=1}^{N} \alpha_T(i)$
    

**Worked Example: Forward Algorithm for $O = (A, B)$**

**1. Initialization ($t=1$, Obs=A):**

- $\alpha_1(1) = \pi_1 \cdot b_1(A) = 0.8 \cdot 0.5 = 0.4$
    
- $\alpha_1(2) = \pi_2 \cdot b_2(A) = 0.2 \cdot 0.2 = 0.04$
    

**2. Induction ($t=2$, Obs=B):**

- For S1 ($j=1$):
    
    $\alpha_2(1) = [\alpha_1(1)a_{11} + \alpha_1(2)a_{21}] \cdot b_1(B)$
    
    $\alpha_2(1) = [ (0.4 \cdot 0.6) + (0.04 \cdot 0.3) ] \cdot 0.5$
    
    $\alpha_2(1) = [0.24 + 0.012] \cdot 0.5 = 0.252 \cdot 0.5 = 0.126$
    
- For S2 ($j=2$):
    
    $\alpha_2(2) = [\alpha_1(1)a_{12} + \alpha_1(2)a_{22}] \cdot b_2(B)$
    
    $\alpha_2(2) = [ (0.4 \cdot 0.4) + (0.04 \cdot 0.7) ] \cdot 0.8$
    
    $\alpha_2(2) = [0.16 + 0.028] \cdot 0.8 = 0.188 \cdot 0.8 = 0.1504$
    

**3. Termination:**

- $P(O | \lambda) = \alpha_2(1) + \alpha_2(2)$
    
- $P(O | \lambda) = 0.126 + 0.1504 = 0.2764$
    

### Practice Problem: Forward Algorithm

**Problem:** Using the same HMM, calculate the total probability $P(O | \lambda)$ for $O = (B, A)$.

Solution:

1. Initialization ($t=1$, Obs=B):

- $\alpha_1(1) = \pi_1 \cdot b_1(B) = 0.8 \cdot 0.5 = 0.4$
    
- $\alpha_1(2) = \pi_2 \cdot b_2(B) = 0.2 \cdot 0.8 = 0.16$
    

**2. Induction ($t=2$, Obs=A):**

- $\alpha_2(1) = [\alpha_1(1)a_{11} + \alpha_1(2)a_{21}] \cdot b_1(A)$
    
    $\alpha_2(1) = [ (0.4 \cdot 0.6) + (0.16 \cdot 0.3) ] \cdot 0.5$
    
    $\alpha_2(1) = [0.24 + 0.048] \cdot 0.5 = 0.288 \cdot 0.5 = 0.144$
    
- $\alpha_2(2) = [\alpha_1(1)a_{12} + \alpha_1(2)a_{22}] \cdot b_2(A)$
    
    $\alpha_2(2) = [ (0.4 \cdot 0.4) + (0.16 \cdot 0.7) ] \cdot 0.2$
    
    $\alpha_2(2) = [0.16 + 0.112] \cdot 0.2 = 0.272 \cdot 0.2 = 0.0544$
    

**3. Termination:**

- $P(O | \lambda) = \alpha_2(1) + \alpha_2(2)$
    
- $P(O | \lambda) = 0.144 + 0.0544 = 0.1984$
    

---

## 4. Algorithm Time Complexity Summary

This table summarizes the computational complexity for the HMM algorithms.

|**Algorithm**|**Problem Type**|**Time Complexity**|**Variables**|
|---|---|---|---|
|**Direct Computation**|Evaluation|$O(T \cdot N^T)$|$N$ = # States, $T$ = Obs. Length|
|**Forward Algorithm**|Evaluation|$O(N^2 T)$|$N$ = # States, $T$ = Obs. Length|
|**Viterbi Algorithm**|Decoding|$O(N^2 T)$|$N$ = # States, $T$ = Obs. Length|

> **Note:** The Forward and Viterbi algorithms, with their $O(N^2 T)$ complexity, offer a massive and essential improvement over the infeasible $O(T \cdot N^T)$ brute-force direct computation.

---
