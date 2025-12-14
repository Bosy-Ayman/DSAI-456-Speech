``` python

def manual_forward(obs_seq, model):

    T = obs_seq.shape[0]
    N = model.n_components

    startprob = model.startprob_
    transmat = model.transmat_
    means = model.means_
    covars = model.covars_

    # Emission probabilities
    emission = np.zeros((T, N))
    for n in range(N):
        emission[:, n] = multivariate_normal.pdf(
            obs_seq, mean=means[n], cov=covars[n]
        )

    alpha = np.zeros((T, N))

    # Initialization
    for j in range(N):
        alpha[0, j] = startprob[j] * emission[0, j]

    # Recursion
    for t in range(1, T):
        for j in range(N):

            # SUM over all previous states
            total = 0
            for i in range(N):
                total += alpha[t-1, i] * transmat[i, j]

            alpha[t, j] = total * emission[t, j]

    # Final probability = sum over last alphas
    return np.sum(alpha[-1])


```

# Viterbi Algorithm

``` python
import numpy as np
from scipy.stats import multivariate_normal

def manual_viterbi(obs_seq, model):

    T = obs_seq.shape[0]        # number of observations
    N = model.n_components      # number of states

    startprob = model.startprob_
    transmat = model.transmat_
    means = model.means_
    covars = model.covars_

    # --------- Emission probabilities ----------
    emission = np.zeros((T, N))
    for n in range(N):
        emission[:, n] = multivariate_normal.pdf(
            obs_seq, mean=means[n], cov=covars[n]
        )

    # --------- Delta and Psi ----------
    delta = np.zeros((T, N))      # best probability to reach state j at time t
    psi = np.zeros((T, N), dtype=int)   # best previous state

    # --------- Initialization ----------
    # δ1(j) = π(j) * b_j(o1)
    delta[0] = startprob * emission[0]

    # --------- Recursion ----------
    # δt(j) = max_i [ δ(t-1)(i) * a(i,j) ] * b_j(ot)
    for t in range(1, T):
        for j in range(N):

            # compute all δ(t-1)(i) * a(i,j)
            probs = delta[t-1] * transmat[:, j]

            # choose best previous state
            best_prev = np.argmax(probs)

            # store new delta
            delta[t, j] = probs[best_prev] * emission[t, j]
            psi[t, j] = best_prev

    # --------- Termination ----------
    best_last_state = np.argmax(delta[-1])

    # --------- Backtracking ----------
    path = [best_last_state]
    for t in range(T-1, 0, -1):
        path.append(psi[t, path[-1]])

    return path[::-1]

```

Here is the step-by-step mathematical explanation for both algorithms.

To understand these algorithms, we first need to define the **Notation** used in Hidden Markov Models (HMM):

- $T$: The length of the observation sequence (time steps).
    
- $N$: The number of hidden states.
    
- $O = (o_1, o_2, ..., o_T)$: The sequence of observations (the `obs_seq`).
    
- $S = \{s_1, ..., s_N\}$: The set of possible hidden states.
    
- $\pi_i$: Initial probability of starting in state $i$ (`startprob_`).
    
- $a_{ij}$: Transition probability from state $i$ to state $j$ (`transmat_`).
    
- $b_j(o_t)$: Emission probability (likelihood) of observing $o_t$ given we are in state $j$.
    

---

### **1. The Emission Probability (Common to Both)**

In both functions, the code starts by pre-calculating the `emission` matrix.

**Code:**

Python

```
emission[:, n] = multivariate_normal.pdf(obs_seq, mean=means[n], cov=covars[n])
```

Math:

Unlike discrete HMMs where you look up a probability in a table, this is a Gaussian HMM. The probability of an observation $o_t$ given state $j$ is calculated using the Probability Density Function (PDF) of a Multivariate Normal distribution:

$$b_j(o_t) = \mathcal{N}(o_t; \mu_j, \Sigma_j)$$

- $\mu_j$: The mean vector for state $j$ (`means[n]`).
    
- $\Sigma_j$: The covariance matrix for state $j$ (`covars[n]`).
    

---

### **2. The Forward Algorithm**

**Goal:** Calculate $P(O|\lambda)$. This is the probability that the model generated this specific sequence of observations.

This algorithm computes the Forward Variable $\alpha_t(j)$, which is defined as:

$$\alpha_t(j) = P(o_1, o_2, ..., o_t, q_t = s_j | \lambda)$$

(The probability of seeing the observations up to time $t$ AND ending up in state $j$ at time $t$).

#### **Step A: Initialization**

**Code:**

Python

```
alpha[0, j] = startprob[j] * emission[0, j]
```

Math:

For the first time step ($t=1$), the probability of being in state $j$ is the prior probability times the likelihood of the first observation:

$$\alpha_1(j) = \pi_j \cdot b_j(o_1)$$

#### **Step B: Recursion (The Inductive Step)**

**Code:**

```Python
total = 0
for i in range(N):
    total += alpha[t-1, i] * transmat[i, j] # Sum over previous states

alpha[t, j] = total * emission[t, j]
```

Math:

To find the probability of being in state $j$ at time $t$, we must sum up the probabilities of coming from all possible previous states $i$:

$$\alpha_t(j) = \left[ \sum_{i=1}^{N} \alpha_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t)$$

1. $\alpha_{t-1}(i)$: Prob of being in state $i$ previously.
    
2. $a_{ij}$: Prob of moving from $i$ to $j$.
    
3. $\sum$: Summing covers all possible paths to reach $j$.
    
4. $b_j(o_t)$: Update with the evidence (observation) at current time $t$.
    

#### **Step C: Termination**

**Code:**

Python

```
return np.sum(alpha[-1])
```

Math:

The total probability of the observation sequence is the sum of the probabilities of ending in any state at time $T$:

$$P(O|\lambda) = \sum_{j=1}^{N} \alpha_T(j)$$

![Image of HMM Forward Algorithm trellis](https://encrypted-tbn3.gstatic.com/licensed-image?q=tbn:ANd9GcTHOVdsXYLlVaZ1nT9Nnk1ub370eEbWfB8QeeL_iaN2ujCt2amS_jfm3ZYHpoeLpb1mS4WEBBT3I8UIYMRUg_IncLbPsqM77vS6DJWM94ZTgBTcMhA)

Shutterstock

Explore

---

### **3. The Viterbi Algorithm**

**Goal:** Find the single "best" path $Q^* = (q_1, q_2, ..., q_T)$. This is the sequence of hidden states that maximizes the likelihood of the observations.

We define the Viterbi Variable $\delta_t(j)$:

$$\delta_t(j) = \max_{q_1...q_{t-1}} P(q_1...q_{t-1}, q_t=s_j, o_1...o_t | \lambda)$$

(The score of the single best path ending in state $j$ at time $t$).

#### **Step A: Initialization**

**Code:**

Python

```
delta[0] = startprob * emission[0]
```

Math:

Identical to the Forward algorithm. The probability of the start of the path:

$$\delta_1(j) = \pi_j \cdot b_j(o_1)$$

#### **Step B: Recursion (Maximization)**

**Code:**


```Python
probs = delta[t-1] * transmat[:, j]  # Calculate transition from all i to j
best_prev = np.argmax(probs)         # Which i gave the max probability?
delta[t, j] = probs[best_prev] * emission[t, j]
psi[t, j] = best_prev
```

Math:

Here is the key difference. Instead of Summing (like Forward), we take the Maximum. We want the best single predecessor.

1. Calculate Max Score ($\delta$):
    
    $$\delta_t(j) = \max_{1 \le i \le N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right] \cdot b_j(o_t)$$
    
2. Keep Track of the Path ($\psi$):
    
    We need to remember which previous state $i$ yielded that maximum value so we can backtrack later.
    
    $$\psi_t(j) = \operatorname*{argmax}_{1 \le i \le N} \left[ \delta_{t-1}(i) \cdot a_{ij} \right]$$
    

#### **Step C: Termination**

**Code:**


```Python
best_last_state = np.argmax(delta[-1])
```

Math:

The score of the best complete path is the max value at time $T$:

$$P^* = \max_{1 \le j \le N} \delta_T(j)$$

The last state of the optimal path ($q_T^*$) is:

$$q_T^* = \operatorname*{argmax}_{1 \le j \le N} [\delta_T(j)]$$

#### **Step D: Backtracking**

**Code:**


```Python
path = [best_last_state]
for t in range(T-1, 0, -1):
    path.append(psi[t, path[-1]])
return path[::-1]
```

Math:

We assume the best path ends at $q_T^*$. We then use the $\psi$ matrix to trace backwards:

$$q_t^* = \psi_{t+1}(q_{t+1}^*)$$

This reconstructs the sequence from $T$ down to $1$.

### **Summary of Differences**

|**Feature**|**Forward (manual_forward)**|**Viterbi (manual_viterbi)**|
|---|---|---|
|**Mathematical Operation**|Summation ($\sum$)|Maximization ($\max$)|
|**Variable Meaning**|Sum of probabilities of _all_ paths reaching state $j$.|Probability of the _single best_ path reaching state $j$.|
|**Output**|A single probability (float).|A sequence of states (list of integers).|
|**Purpose**|Parameter learning, Sequence Likelihood.|Decoding (finding what happened underneath).|
