
### HMM for Phoneme Recognition: Problem

Consider a Hidden Markov Model ($\lambda$) for recognizing a sequence of phonemes. The model has two hidden states ($N=2$) corresponding to the phonemes **`/k/` (state 1)** and **`/æ/` (state 2)**.

The model parameters (initial probabilities $\pi_i$, transition probabilities $a_{ij}$, and emission probabilities $b_j(o_t)$) are given by the following tables. The observations $o_t$ are discrete acoustic features from the set $\{v_1, v_2\}$.

**1. Initial State Probabilities (**$\pi_i$**)** This is the probability of starting in a given phoneme state. $P(q_1 = i)$

|$i$ (Phoneme)|$\pi_i$|
|---|---|
|1: /k/|0.6|
|2: /æ/|0.4|

**2. State Transition Probabilities (**$a_{ij}$**)** This is the probability of moving from one phoneme to another. $P(q_{t+1} = j \mid q_t = i)$

|$q_t = i$|$q_{t+1} = j$|$a_{ij}$|
|---|---|---|
|1: /k/|1: /k/|0.5|
|1: /k/|2: /æ/|0.5|
|2: /æ/|1: /k/|0.3|
|2: /æ/|2: /æ/|0.7|

**3. Observation Emission Probabilities (**$b_j(o_t)$**)** This is the probability of observing an acoustic feature given the current phoneme. $P(o_t \mid q_t = j)$

|$q_t = j$|$o_t$|$b_j(o_t)$|
|---|---|---|
|1: /k/|$v_1$|0.8|
|1: /k/|$v_2$|0.2|
|2: /æ/|$v_1$|0.1|
|2: /æ/|$v_2$|0.9|

### ❓ Question

Suppose we observe the acoustic feature sequence $O = (o_1, o_2)$, where $o_1 = v_1$ and $o_2 = v_2$.

Using the **Forward Algorithm**, compute the total likelihood of this observation sequence, $P(O \mid \lambda)$, one step at a time.

**(a) Initialization:** Compute the forward probabilities $\alpha_1(i)$ for all states $i$ at time $t=1$.

**(b) Recursion:** Compute the forward probabilities $\alpha_2(j)$ for all states $j$ at time $t=2$.

**(c) Termination:** Compute the final likelihood $P(O \mid \lambda)$.

### 💡 Model Answer & Solution

Here is the step-by-step computation using the Forward Algorithm.

#### (a) Initialization (t=1)

We compute the forward probability for each state $i$ at time $t=1$. The formula is: $\alpha_1(i) = \pi_i b_i(o_1)$ The first observation is $o_1 = v_1$.

- **For State 1 (/k/):** $\alpha_1(1) = \pi_1 \times b_1(v_1) = 0.6 \times 0.8 = 0.48$
    
- **For State 2 (/æ/):** $\alpha_1(2) = \pi_2 \times b_2(v_1) = 0.4 \times 0.1 = 0.04$
    

**Answer (a):** $\alpha_1(1) = 0.48$ **and** $\alpha_1(2) = 0.04$**.** This represents the probability of being in each state _and_ having seen the first observation $v_1$.

#### (b) Recursion (t=2)

We compute the forward probability for each state $j$ at time $t=2$. The formula is: $\alpha_2(j) = \left( \sum_{i=1}^N \alpha_1(i) a_{ij} \right) b_j(o_2)$ The second observation is $o_2 = v_2$.

- **For State 1 (/k/):**
    
    1. First, sum the probabilities of transitioning _to_ state 1 from all possible previous states: $\sum_{i=1}^2 \alpha_1(i) a_{i1} = [\alpha_1(1) \times a_{11}] + [\alpha_1(2) \times a_{21}]$ $= [0.48 \times 0.5] + [0.04 \times 0.3]$ $= 0.24 + 0.012 = 0.252$
        
    2. Then, multiply by the emission probability for state 1 and observation $o_2$: $\alpha_2(1) = (0.252) \times b_1(v_2) = 0.252 \times 0.2 = 0.0504$
        
- **For State 2 (/æ/):**
    
    1. First, sum the probabilities of transitioning _to_ state 2 from all possible previous states: $\sum_{i=1}^2 \alpha_1(i) a_{i2} = [\alpha_1(1) \times a_{12}] + [\alpha_1(2) \times a_{22}]$ $= [0.48 \times 0.5] + [0.04 \times 0.7]$ $= 0.24 + 0.028 = 0.268$
        
    2. Then, multiply by the emission probability for state 2 and observation $o_2$: $\alpha_2(2) = (0.268) \times b_2(v_2) = 0.268 \times 0.9 = 0.2412$
        

**Answer (b):** $\alpha_2(1) = 0.0504$ **and** $\alpha_2(2) = 0.2412$**.**

#### (c) Termination

We compute the final likelihood by summing the forward probabilities at the final time step $T=2$. The formula is: $P(O \mid \lambda) = \sum_{j=1}^N \alpha_T(j)$

$P(O \mid \lambda) = \alpha_2(1) + \alpha_2(2)$ $P(O \mid \lambda) = 0.0504 + 0.2412 = 0.2916$

**Answer (c): The total likelihood of observing the sequence** $O = (v_1, v_2)$ **given the model** $\lambda$ **is 0.2916.**