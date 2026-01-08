Excellent! Now we're tackling **Problem 2: Finding the Best State Sequence** using the **Viterbi Algorithm**. This is one of the most important algorithms in HMMs.

## The Goal

**Find the single best hidden state sequence Q that most likely produced the observations O.**

We want to maximize: **P(Q|O, λ)**

Which is equivalent to maximizing: **P(Q, O|λ)**

---

## The Viterbi Variable: δₜ(i)

**δₜ(i) = max_{q₁,q₂,...,qₜ₋₁} P(q₁q₂...qₜ₋₁, qₜ=Sᵢ, O₁O₂...Oₜ | λ)**

This is:
- The **highest probability** of any state path
- That ends in state Sᵢ at time t
- And accounts for observations O₁...Oₜ

**Key difference from Forward:**
- **Forward α**: **SUMS** over all possible paths
- **Viterbi δ**: Takes the **MAX** over all possible paths

---

## The Key Insight: Why MAX Instead of SUM?

**Forward algorithm:**
"What's the total probability of all paths that could produce O?"
→ Sum all possibilities

**Viterbi algorithm:**
"What's the single most likely path that produced O?"
→ Keep only the best path at each step

---

## The Viterbi Algorithm Steps

### Step 1: Initialization (t=1)

**δ₁(i) = πᵢ · bᵢ(O₁)** for all states i
**ψ₁(i) = 0** (no previous state at t=1)

Same as Forward initialization!

**Example:**
```
O = (H, T)
δ₁(1) = 0.6 × 0.5 = 0.30
δ₁(2) = 0.4 × 0.9 = 0.36
```

---

### Step 2: Recursion (t=2 to T)

**δₜ(j) = max_{1≤i≤N} [δₜ₋₁(i) · aᵢⱼ] · bⱼ(Oₜ)**

**ψₜ(j) = argmax_{1≤i≤N} [δₜ₋₁(i) · aᵢⱼ]**

**Breaking this down:**
1. **δₜ₋₁(i) · aᵢⱼ**: Best path to state i at t-1, then transition to j
2. **max**: Choose the BEST previous state (not sum over all)
3. **bⱼ(Oₜ)**: Multiply by emission probability
4. **ψₜ(j)**: Remember which previous state was best (for backtracking)

---

### Step 3: Termination

**P* = max_{1≤i≤N} [δₜ(i)]** (best final probability)

**q*ₜ = argmax_{1≤i≤N} [δₜ(i)]** (best final state)

---

### Step 4: Backtracking

**q*ₜ = ψₜ₊₁(q*ₜ₊₁)** for t = T-1, T-2, ..., 1

Trace back through the ψ array to reconstruct the optimal path.

---

## Complete Numerical Example

Using our model with **O = (H, T)**:

**Model:**
```
States: S₁ (fair), S₂ (biased)
π₁ = 0.6, π₂ = 0.4
a₁₁ = 0.7, a₁₂ = 0.3, a₂₁ = 0.4, a₂₂ = 0.6
b₁(H) = 0.5, b₁(T) = 0.5, b₂(H) = 0.9, b₂(T) = 0.1
```

---

### **Step 1: Initialization (t=1, O₁=H)**

```
δ₁(1) = π₁ × b₁(H) = 0.6 × 0.5 = 0.30
δ₁(2) = π₂ × b₂(H) = 0.4 × 0.9 = 0.36

ψ₁(1) = 0
ψ₁(2) = 0
```

---

### **Step 2: Recursion (t=2, O₂=T)**

**For δ₂(1):** (reaching S₁ at time 2)

We need to find the MAX:

**Option A: Coming from S₁**
```
δ₁(1) × a₁₁ = 0.30 × 0.7 = 0.21
```

**Option B: Coming from S₂**
```
δ₁(2) × a₂₁ = 0.36 × 0.4 = 0.144
```

**Take the MAX:**
```
max(0.21, 0.144) = 0.21 ← came from S₁
```

**Multiply by emission:**
```
δ₂(1) = 0.21 × b₁(T) = 0.21 × 0.5 = 0.105
```

**Remember which state was best:**
```
ψ₂(1) = argmax = 1 (came from S₁)
```

---

**For δ₂(2):** (reaching S₂ at time 2)

**Option A: Coming from S₁**
```
δ₁(1) × a₁₂ = 0.30 × 0.3 = 0.09
```

**Option B: Coming from S₂**
```
δ₁(2) × a₂₂ = 0.36 × 0.6 = 0.216
```

**Take the MAX:**
```
max(0.09, 0.216) = 0.216 ← came from S₂
```

**Multiply by emission:**
```
δ₂(2) = 0.216 × b₂(T) = 0.216 × 0.1 = 0.0216
```

**Remember which state was best:**
```
ψ₂(2) = argmax = 2 (came from S₂)
```

---

### **Step 3: Termination (t=2=T)**

```
P* = max[δ₂(1), δ₂(2)]
   = max[0.105, 0.0216]
   = 0.105

q*₂ = argmax[δ₂(1), δ₂(2)]
    = 1 (S₁)
```

**The best path ends in state S₁ with probability 0.105**

---

### **Step 4: Backtracking**

```
q*₂ = 1 (we just found this)

q*₁ = ψ₂(q*₂)
    = ψ₂(1)
    = 1
```

**Optimal state sequence: Q* = (S₁, S₁)**

**Interpretation:** "The most likely explanation for observations (H, T) is that we used the fair coin (S₁) for both flips."

---

## Visual Comparison: Forward vs Viterbi

### Forward Algorithm (at t=2):
```
         S₁(0.30)────0.21────┐
              ↘              ↓ SUM
                ─────0.144───┤
         S₂(0.36)────0.216───┘
                              ↓
                         α₂(1) = 0.177
```
Sums: 0.21 + 0.144 = 0.354 (before emission)

### Viterbi Algorithm (at t=2):
```
         S₁(0.30)────0.21────┐
              ↘     (WIN!)   ↓ MAX
                ─────0.144───┤V
         S₂(0.36)────0.216───┘
                              ↓
                         δ₂(1) = 0.105
```
Max: max(0.21, 0.144) = 0.21 (before emission)

---

## The Trellis Diagram

```
Time:      t=1              t=2
          ┌────┐           ┌────┐
State S₁: │0.30│──0.21────→│0.105│ ← BEST PATH
          └────┘     ↖     └────┘
                      ╲
          ┌────┐      ╲   ┌────┐
State S₂: │0.36│──0.144──→│0.022│
          └────┘           └────┘
          
Obs:       H                T

The thick line shows the optimal path: S₁ → S₁
```

---

## Complete Example with Longer Sequence

Let's do **O = (H, T, H)** to show backtracking better:

After running the algorithm through t=3:
```
δ₃(1) = 0.0315  ← BEST
δ₃(2) = 0.0057

ψ₃(1) = 1 (came from S₁)
ψ₃(2) = 1 (came from S₁)
```

**Step 3: Termination**
```
q*₃ = 1 (S₁ is best at t=3)
```

**Step 4: Backtracking**
```
q*₃ = 1
q*₂ = ψ₃(q*₃) = ψ₃(1) = 1
q*₁ = ψ₂(q*₂) = ψ₂(1) = 1
```

**Optimal path: Q* = (S₁, S₁, S₁)**

"Most likely used the fair coin for all three flips."

---

## Key Differences Summary

| Algorithm | Operation | Purpose | Result |
|-----------|-----------|---------|---------|
| **Forward** | SUM | Total probability | P(O\|λ) = 0.177 |
| **Viterbi** | MAX | Best single path | Q* = (S₁,S₁), P*=0.105 |

**Important:** P* ≤ P(O|λ) always!
- P* is just ONE path's probability
- P(O|λ) is the sum over ALL paths

---

## Why Viterbi is Useful

**Applications:**
- **Speech recognition**: "What sequence of phonemes most likely produced this audio?"
- **POS tagging**: "What sequence of parts-of-speech best explains this sentence?"
- **Bioinformatics**: "What gene structure best explains this DNA sequence?"
- **Error correction**: "What message was most likely sent given this noisy signal?"

The Viterbi algorithm finds the single best explanation, which is often what we want for decoding/recognition tasks!

Does this make sense?