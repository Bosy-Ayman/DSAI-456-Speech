This explanation maps the Python code directly to the mathematical definitions found in the original CTC paper (Graves et al.).

We define the extended target sequence (with blanks) as $y'$ (prime), where indices $s$ go from $0$ to $|y'|$. We define the input sequence $x$ over time $t$.

### 1\. The Variables

  * **$\alpha_t(s)$**: The probability of the valid path prefix ending at state $s$ at time $t$. (Represented by `alphas[t, s]`).
  * **$y'_s$**: The character at index $s$ in the extended target. (Represented by `extended_target[s]`).
  * **$P(k|t)$**: The probability of character $k$ at time $t$ output by the neural net. (Represented by `logits[t, char]`).

-----

### 2\. Initialization ($t=0$)

In CTC, a valid path must start at the very beginning.

**Code:**

```python
alphas[0, 0] = logits[0, self.blank]
if L > 0:
    alphas[0, 1] = logits[0, target[0]]
```

**Equation:**

$$
\alpha_1(0) = P(\text{blank} | t=1) \\

\alpha_1(1) = P(y'_1 | t=1) \\

\alpha_1(s) = 0, \quad \forall s > 1
$$

**Explanation:**
You can only start at the first **Blank** ($s=0$) or the first **Character** ($s=1$). Starting anywhere else ($s>1$) is impossible, so probability is 0.

-----

### 3\. The Recursive Step ($t > 0$)

This is the core "Forward Algorithm." We calculate $\alpha_t(s)$ based on $\alpha_{t-1}$.

#### Case A: Standard Transition (Stay or Step)

This applies when we are at a **Blank**, or when the current character is the **same as the previous character** (e.g., the second 'L' in 'HELLO').

**Code:**

```python
# Path 1 (Stay) + Path 2 (Step)
alpha_sum = alphas[t - 1, s]
if s > 0:
    alpha_sum += alphas[t - 1, s - 1]
    
# Multiply by probability of current char
alphas[t, s] = alpha_sum * logits[t, int(char)]
```

**Equation:**

$$
\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) \right) \cdot P(y'_s | t)
$$

  * $\alpha_{t-1}(s)$: **Stay** (Self-loop).
  * $\alpha_{t-1}(s-1)$: **Step** (Transition from previous).

#### Case B: The "Skip" Transition

This applies when we are at a **Character**, it is **not a blank**, and it is **different from the previous character**.

**Code:**

```python
# Path 3 (Skip)
if s > 1 and extended_target[s] != self.blank and \
   extended_target[s] != extended_target[s - 2]:
    alpha_sum += alphas[t - 1, s - 2]
```

**Equation:**

$$
\alpha_t(s) = \left( \alpha_{t-1}(s) + \alpha_{t-1}(s-1) + \mathbf{\alpha_{t-1}(s-2)} \right) \cdot P(y'_s | t)
$$

**Explanation:**

  * We add the third term $\alpha_{t-1}(s-2)$.
  * This represents jumping over the blank that sits between $s$ and $s-2$.

-----

### 4\. Total Probability & Loss

After processing all time steps up to $T$, we need the probability of the entire sequence.

**Code:**

```python
# Sum over both possible final positions (Last char OR Final Blank)
prob = alphas[t_len - 1, 2 * y_len - 1] + alphas[t_len - 1, 2 * y_len]
loss = -np.log(prob)
```

**Equation:**

$$
P(y|x) = \alpha_T(|y'|) + \alpha_T(|y'|-1)
$$

$$
\mathcal{L}(x, y) = -\ln(P(y|x))
$$

**Explanation:**

  * $|y'|$ is the index of the final blank.
  * $|y'|-1$ is the index of the final character.
  * A valid sentence is finished if we end on the last letter **OR** if we finish the last letter and then move to the final blank. We sum both probabilities.
  * We take the **Negative Log Likelihood** to turn it into a loss function (since gradient descent minimizes loss, but we want to maximize probability).

### Summary of the Mapping

| Concept                     | Code               | Math Term           |
| :-------------------------- | :----------------- | :------------------ |
| **Current Probability**     | `logits[t, char]`  | $P(y'_s \mid t)$    |
| **Accumulated Probability** | `alphas[t, s]`     | $\alpha_t(s)$       |
| **Stay**                    | `alphas[t-1, s]`   | $\alpha_{t-1}(s)$   |
| **Step**                    | `alphas[t-1, s-1]` | $\alpha_{t-1}(s-1)$ |
| **Skip**                    | `alphas[t-1, s-2]` | $\alpha_{t-1}(s-2)$ |

---

This `CTCDecoder` class converts the raw probability matrix (`logits`) from the neural network into the final human-readable sequence (e.g., "HELLO").

It implements two different strategies: **Greedy Decoding** (fast but less accurate) and **Beam Search** (slower but finds better sentences).

---

### Part 1: Greedy Decoding

**Method:** `greedy_decode`

This is the simplest approach. It trusts the highest probability at every single millisecond, without looking ahead.

#### **How it works (The Logic)**

1. **Argmax:** For every time step $t$, pick the class with the highest probability.
    
2. **Collapse:** Simplify the sequence using CTC rules:
    
    - Ignore **Blanks**.
        
    - Ignore **Duplicates** (only if they are consecutive and not separated by a blank).
        

#### **Numerical Trace**

Imagine the `argmax` output over 5 frames is: `[A, A, -, -, B]`

- **Step 1:** `p=A`. Not blank. Not prev (`-`). **Keep A**. (Prev is now A).
    
- **Step 2:** `p=A`. Not blank. `p == prev` (A). **Ignore**.
    
- **Step 3:** `p=-`. Is blank. **Ignore**. (Prev is now -).
    
- **Step 4:** `p=-`. Is blank. **Ignore**.
    
- **Step 5:** `p=B`. Not blank. Not prev (`-`). **Keep B**.
    
- **Result:** `[A, B]`
    

---

### Part 2: Beam Search Decoding

**Method:** `beam_search_decode`

Greedy decoding fails if the best individual letters don't form the most likely _word_. Beam search keeps the **Top K** (beam width) most likely paths active at every step, creating a "tree" of possibilities, and prunes the bad ones.

#### **The Critical Concept: `pb` vs `pnb`**

In the code, every beam (candidate sequence) stores two scores:

- **`pb` (Probability ending in Blank):** The score of this path if the _very last_ symbol processed was a blank.
    
- **`pnb` (Probability ending in Non-Blank):** The score of this path if the _very last_ symbol was a character.
    

Why separate them?

To handle the "Double Letter" rule.

- If we have "A" and the next letter is "A":
    
    - If "A" ended in a Blank (`A -`), adding `A` creates **"AA"**.
        
    - If "A" ended in a Char (A), adding A merges into "A".
        
        We cannot know which rule to apply unless we track pb and pnb separately.
        

#### **Code Logic Trace**

**1. Initialization**

Python

```
beams = {("", self.blank): (0, 0)}
```

We start with one beam: an empty string. `last_char` is blank. Scores are 0.

2. The Loop (Time $t$)

For every active beam, we try adding every possible character c from the vocabulary.

**Case A: Extending with a Blank (`c == blank`)**

Python

```
new_pb = pb + prob  # Log-prob addition (which is multiplication)
```

- We only update `new_pb`. The sequence text (`seq`) doesn't change.
    
- This path represents "staying in silence" or "transitioning to silence."
    

Case B: Extending with a Character (c != blank)

Here, we have branching logic based on the last_char of the current beam.

- **Scenario 1: Same Character (`c == last_char`)**
    
    - _Example:_ We have "A", we see "A".
        
    - `new_pnb = pnb + prob`
        
    - We effectively "merge" the repeated character. The text remains "A", but the score improves.
        
- **Scenario 2: Different Character (`c != last_char`)**
    
    - _Example:_ We have "A", we see "B".
        
    - `new_pnb = pb + pnb + prob`
        
    - We sum the probabilities of previous blank paths and non-blank paths (because both `A - B` and `AB` lead to the same result here), then multiply by probability of `B`.
        
    - **Result:** New sequence is "AB".
        

3. Merging Paths (if key not in new_beams...)

Sometimes, two different paths lead to the same result.

- Path 1: `A - A` $\to$ Result "AA"
    
- Path 2: `A A` (distinct As) $\to$ Result "AA"
    
- Beam search sums the probabilities of these converging paths:
    
    Python
    
    ```
    new_beams[key] = (old_pb + new_pb, old_pnb)
    ```
    

**4. Pruning (The "Beam" part)**

Python

```
scored.sort(key=lambda x: x[1], reverse=True)
beams = ... [:beam_width]
```

At the end of every time step, we might have thousands of branches. We sort them by score and **kill** everything except the top `beam_width` (e.g., top 50). This keeps memory usage constant.

### Summary: Greedy vs. Beam

|**Feature**|**Greedy Decode**|**Beam Search**|
|---|---|---|
|**Speed**|Very Fast|Slower (depends on `beam_width`)|
|**Accuracy**|Lower (can make spelling errors)|Higher (considers context)|
|**Lookahead**|None (1 step at a time)|Yes (maintains multiple futures)|
|**Complexity**|$O(T)$|$O(T \times BeamWidth \times Classes)$|