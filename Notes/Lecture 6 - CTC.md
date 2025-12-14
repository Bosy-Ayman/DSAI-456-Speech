This is a cleaned-up, step-by-step derivation of CTC Loss.

To make this clearer than your previous example, we will explicitly define the **Transition Rules** (Stay, Step, Skip) and use probabilities that highlight the "correct" path, so the math is easier to follow.

### **The Setup**

- **Target Sequence:** $Y = [A, B]$
    
- **Extended Target ($Y'$):** `[-, A, -, B, -]`
    
    - Indices ($s$): $0, 1, 2, 3, 4$
        
- **Time Frames ($T$):** 3
    
- **Input Probabilities (Logits):**
    
    - We want the model to predict A, then transition to B.
        
    - _Note: Rows sum to 1.0_
        

|**Frame (t)**|**Blank (-)**|**A**|**B**|
|---|---|---|---|
|**0**|0.2|**0.8**|0.0|
|**1**|0.4|0.2|**0.4**|
|**2**|0.1|0.0|**0.9**|

---

### **The Three Transition Rules**

When calculating probability for state $s$ at time $t$, we can arrive from previous states at $t-1$:

1. **Stay:** From same state $s$ (e.g., $A \to A$).
    
2. **Step:** From previous state $s-1$ (e.g., $- \to A$).
    
3. **Skip:** From state $s-2$ (e.g., $A \to B$).
    
    - _Condition:_ You can only skip the blank if the current char and the char 2 steps back are distinct (i.e., $A \neq B$). Since our target is $[A, B]$, skips are allowed.
        

---

### **Step 1: Initialization ($t=0$)**

In CTC, we can start only at the first **Blank** ($s=0$) or the first **Character** ($s=1$).

- $\alpha_{0}(0) \text{ [Blank]} = P(t=0, \text{-}) = \mathbf{0.2}$
    
- $\alpha_{0}(1) \text{ [A]} = P(t=0, A) = \mathbf{0.8}$
    
- All others = 0.
    

**Alpha Vector at $t=0$:** `[0.2, 0.8, 0, 0, 0]`

---

### **Step 2: Forward Pass ($t=1$)**

**Current Probabilities:** $P(-)=0.4, P(A)=0.2, P(B)=0.4$

- $s=0$ (Blank): Can only come from $s=0$ (Stay).
    
    $$\alpha_{1}(0) = \alpha_{0}(0) \cdot P(-) = 0.2 \cdot 0.4 = \mathbf{0.08}$$
    
- $s=1$ (A): From $s=1$ (Stay) or $s=0$ (Step).
    
    $$\alpha_{1}(1) = (\alpha_{0}(1) + \alpha_{0}(0)) \cdot P(A)$$
    
    $$\alpha_{1}(1) = (0.8 + 0.2) \cdot 0.2 = \mathbf{0.2}$$
    
- $s=2$ (Blank): From $s=2$ (Stay - impossible, was 0) or $s=1$ (Step).
    
    $$\alpha_{1}(2) = (\alpha_{0}(2) + \alpha_{0}(1)) \cdot P(-)$$
    
    $$\alpha_{1}(2) = (0 + 0.8) \cdot 0.4 = \mathbf{0.32}$$
    
- **$s=3$ (B):** From $s=3$ (Stay), $s=2$ (Step), or **$s=1$ (Skip)**.
    
    - Skip Check: Can we jump $A \to B$? Yes, distinct characters.
        
        $$\alpha_{1}(3) = (\alpha_{0}(3) + \alpha_{0}(2) + \alpha_{0}(1)) \cdot P(B)$$
        
        $$\alpha_{1}(3) = (0 + 0 + 0.8) \cdot 0.4 = \mathbf{0.32}$$
        
- **$s=4$ (Blank):** Too far to reach yet. $\mathbf{0}$.
    

**Alpha Vector at $t=1$:** `[0.08, 0.2, 0.32, 0.32, 0]`

---

### **Step 3: Forward Pass ($t=2$)**

**Current Probabilities:** $P(-)=0.1, P(A)=0.0, P(B)=0.9$

- $s=2$ (Blank between A and B): From $s=2$ (Stay) or $s=1$ (Step).
    
    $$\alpha_{2}(2) = (\alpha_{1}(2) + \alpha_{1}(1)) \cdot P(-)$$
    
    $$\alpha_{2}(2) = (0.32 + 0.2) \cdot 0.1 = \mathbf{0.052}$$
    
- **$s=3$ (B):** From $s=3$ (Stay), $s=2$ (Step), or **$s=1$ (Skip)**.
    
    - Note: We sum all valid paths leading to B.
        
        $$\alpha_{2}(3) = (\alpha_{1}(3) + \alpha_{1}(2) + \alpha_{1}(1)) \cdot P(B)$$
        
        $$\alpha_{2}(3) = (0.32 + 0.32 + 0.2) \cdot 0.9 = 0.84 \cdot 0.9 = \mathbf{0.756}$$
        
- $s=4$ (Final Blank): From $s=4$ (Stay) or $s=3$ (Step).
    
    $$\alpha_{2}(4) = (\alpha_{1}(4) + \alpha_{1}(3)) \cdot P(-)$$
    
    $$\alpha_{2}(4) = (0 + 0.32) \cdot 0.1 = \mathbf{0.032}$$
    

**Alpha Vector at $t=2$:** `[..., ..., 0.052, 0.756, 0.032]`

---

### **Step 4: Compute Total Probability & Loss**

A valid CTC path must end at the **last character** ($s=3$, 'B') or the **final blank** ($s=4$).

1. Total Probability $P(Y|X)$:
    
    $$P(Y|X) = \alpha_{2}(3) + \alpha_{2}(4)$$
    
    $$P(Y|X) = 0.756 + 0.032 = \mathbf{0.788}$$
    
2. CTC Loss:
    
    $$Loss = -\ln(P(Y|X))$$
    
    $$Loss = -\ln(0.788) \approx \mathbf{0.238}$$
    

### **Summary of Results**

Because the input probabilities strongly favored the sequence `A -> B` (high probabilities on the diagonal path), the Total Probability is high (0.788) and the Loss is low (0.238).

- **User's Example:** Had aggressive probability drops, resulting in a loss of ~4.02.
    
- **This Example:** Followed a likely path, resulting in a loss of ~0.24.
    

Would you like to see how the Backpropagation (gradients) would be calculated for this specific example?