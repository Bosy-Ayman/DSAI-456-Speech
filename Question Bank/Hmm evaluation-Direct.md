# Beam Search for CTC Decoding

Let me explain beam search step by step with a clear example.

## The Problem

After your network outputs probabilities, you need to **decode** them to get the final text. CTC forward algorithm computes the probability of a specific target (like "DARK"), but for **decoding**, you don't know the target—you need to find the most likely sequence.

---

## Greedy Decoding (Simple but Not Optimal)

**Greedy approach:** At each time step, pick the character with highest probability.
```python
Time:  0    1    2    3    4    5    6
Pick:  D    D    -    A    A    -    R  ...
       ↓    ↓    ↓    ↓    ↓    ↓    ↓
     0.6  0.7  0.8  0.6  0.7  0.5  0.6

After collapse: D - A - R → "DAR"
```

**Problem:** This is locally optimal but might miss the globally best path!

---

## Beam Search (Better Approach)

**Beam search:** Keep track of the **top K most likely sequences** (K = beam width) at each time step, instead of just one.

### Key Idea
- Don't commit to a single path
- Maintain multiple candidate sequences
- At each step, expand all candidates and keep the best K

---

## Step-by-Step Example

**Setup:**
- Vocabulary: `{A, B, C, -}` (3 letters + blank)
- Beam width: K = 2 (keep top 2 sequences)
- Target: Decode the most likely word

### Network Output (Probabilities)
```
Time    A     B     C    blank
t=0   0.1   0.2   0.1    0.6
t=1   0.4   0.1   0.2    0.3
t=2   0.1   0.6   0.1    0.2
t=3   0.2   0.1   0.5    0.2
```

---

## Beam Search Execution

### **t=0: Initialize**

Start with empty sequence `""` with probability 1.0
```
Beam:
Sequence: ""    Prob: 1.0    Last: None
```

Expand by trying all 4 characters:
```
Candidates:
""→A:  prob = 1.0 × 0.1 = 0.1,  collapse to "A"
""→B:  prob = 1.0 × 0.2 = 0.2,  collapse to "B"
""→C:  prob = 1.0 × 0.1 = 0.1,  collapse to "C"
""→-:  prob = 1.0 × 0.6 = 0.6,  collapse to ""
```

**Keep top 2 (beam width = 2):**
```
Beam after t=0:
1. Sequence: ""    Prob: 0.6    Last: -
2. Sequence: "B"   Prob: 0.2    Last: B
```

---

### **t=1: Expand Both Sequences**

#### Expand sequence 1: `""` (last = -)
```
""→A:  prob = 0.6 × 0.4 = 0.24,  collapse to "A"
""→B:  prob = 0.6 × 0.1 = 0.06,  collapse to "B"
""→C:  prob = 0.6 × 0.2 = 0.12,  collapse to "C"
""→-:  prob = 0.6 × 0.3 = 0.18,  collapse to ""
```

#### Expand sequence 2: `"B"` (last = B)
```
"B"→A:  prob = 0.2 × 0.4 = 0.08,  collapse to "BA"
"B"→B:  prob = 0.2 × 0.1 = 0.02,  collapse to "B"  (repeated B)
"B"→C:  prob = 0.2 × 0.2 = 0.04,  collapse to "BC"
"B"→-:  prob = 0.2 × 0.3 = 0.06,  collapse to "B"
```

**All candidates:**
```
"A":   0.24
"":    0.18
"C":   0.12
"BA":  0.08
"B":   0.06 + 0.02 = 0.08  (merge same sequences)
"BC":  0.04
```

**Keep top 2:**
```
Beam after t=1:
1. Sequence: "A"    Prob: 0.24    Last: A
2. Sequence: ""     Prob: 0.18    Last: -
```

---

### **t=2: Expand Again**

#### Expand sequence 1: `"A"` (last = A)
```
"A"→A:  prob = 0.24 × 0.1 = 0.024,  collapse to "A"  (repeated A)
"A"→B:  prob = 0.24 × 0.6 = 0.144,  collapse to "AB"
"A"→C:  prob = 0.24 × 0.1 = 0.024,  collapse to "AC"
"A"→-:  prob = 0.24 × 0.2 = 0.048,  collapse to "A"
```

#### Expand sequence 2: `""` (last = -)
```
""→A:  prob = 0.18 × 0.1 = 0.018,  collapse to "A"
""→B:  prob = 0.18 × 0.6 = 0.108,  collapse to "B"
""→C:  prob = 0.18 × 0.1 = 0.018,  collapse to "C"
""→-:  prob = 0.18 × 0.2 = 0.036,  collapse to ""
```

**All candidates:**
```
"AB":  0.144
"B":   0.108
"A":   0.024 + 0.048 + 0.018 = 0.090  (merge)
"":    0.036
"AC":  0.024
"C":   0.018
```

**Keep top 2:**
```
Beam after t=2:
1. Sequence: "AB"   Prob: 0.144   Last: B
2. Sequence: "B"    Prob: 0.108   Last: B
```

---

### **t=3: Final Step**

#### Expand sequence 1: `"AB"` (last = B)
```
"AB"→A:  prob = 0.144 × 0.2 = 0.0288,  collapse to "ABA"
"AB"→B:  prob = 0.144 × 0.1 = 0.0144,  collapse to "AB"
"AB"→C:  prob = 0.144 × 0.5 = 0.0720,  collapse to "ABC"
"AB"→-:  prob = 0.144 × 0.2 = 0.0288,  collapse to "AB"
```

#### Expand sequence 2: `"B"` (last = B)
```
"B"→A:  prob = 0.108 × 0.2 = 0.0216,  collapse to "BA"
"B"→B:  prob = 0.108 × 0.1 = 0.0108,  collapse to "B"
"B"→C:  prob = 0.108 × 0.5 = 0.0540,  collapse to "BC"
"B"→-:  prob = 0.108 × 0.2 = 0.0216,  collapse to "B"
```

**All candidates:**
```
"ABC":  0.0720  ← Best!
"BC":   0.0540
"AB":   0.0288 + 0.0144 = 0.0432
"ABA":  0.0288
"B":    0.0108 + 0.0216 = 0.0324
"BA":   0.0216
```

**Keep top 2:**
```
Final Beam:
1. Sequence: "ABC"  Prob: 0.0720  ← Winner!
2. Sequence: "BC"   Prob: 0.0540
```

---

## Final Result

**Best decoded sequence: "ABC"** with probability 0.0720

---

## Visual Summary
```
t=0          t=1          t=2          t=3
 
""(0.6)  →  "A"(0.24) → "AB"(0.14) → "ABC"(0.072) ✓ Best!
  ↓          ↓           ↓
"B"(0.2)  → ""(0.18)  → "B"(0.11)  → "BC"(0.054)
```

At each step, we keep the top K sequences (beam width = 2).

---

## Key Points

1. **Beam width K**: Number of sequences to keep
   - K=1: Greedy decoding (fastest, least accurate)
   - K=10-100: Better accuracy, slower
   
2. **Prefix merging**: Sequences that collapse to the same text are merged

3. **CTC rules apply**:
   - Blanks don't extend sequence
   - Repeated characters without blank stay as one

4. **Probability**: Multiply probabilities along the path (in log space: add)

5. **Final choice**: The sequence with highest probability wins

---

## Beam Width Comparison
```
Beam Width = 1 (Greedy):
t=0: Keep only ""
t=1: Keep only "A"
t=2: Keep only "AB"
Result: "ABC" (might miss better paths)

Beam Width = 3:
t=0: Keep "", "B", "A"
t=1: Keep "A", "", "B"
t=2: Keep "AB", "B", "A"
Result: More exploration, better result!
```

**Beam search balances exploration (multiple paths) with efficiency (only top K).**

---

## Python Implementation
```python
import numpy as np

def ctc_beam_search(log_probs, beam_width=10, blank_id=26):
    """
    Beam search decoder for CTC.
    
    Args:
        log_probs: (T, num_classes) - log probabilities from network
        beam_width: number of beams to keep
        blank_id: index of blank token
    
    Returns:
        best_path: decoded sequence (list of character indices)
        best_score: log probability of best path
    """
    T, num_classes = log_probs.shape
    
    # Initialize beam with empty sequence
    # beam format: {sequence: (log_prob, last_char)}
    beam = {(): (0.0, None)}
    
    for t in range(T):
        new_beam = {}
        
        # Expand each sequence in current beam
        for seq, (seq_prob, last_char) in beam.items():
            
            # Try all possible characters
            for c in range(num_classes):
                prob = log_probs[t, c]
                
                if c == blank_id:
                    # Blank: keep same sequence
                    new_seq = seq
                    new_last = last_char
                else:
                    # Non-blank character
                    if c == last_char:
                        # Same character as last: keep same sequence
                        new_seq = seq
                        new_last = c
                    else:
                        # Different character: extend sequence
                        new_seq = seq + (c,)
                        new_last = c
                
                # Calculate new probability
                new_prob = seq_prob + prob
                
                # Add to new beam or update if better
                if new_seq not in new_beam:
                    new_beam[new_seq] = (new_prob, new_last)
                else:
                    # Keep the path with higher probability
                    if new_prob > new_beam[new_seq][0]:
                        new_beam[new_seq] = (new_prob, new_last)
        
        # Keep only top beam_width sequences
        beam = dict(sorted(new_beam.items(), 
                          key=lambda x: x[1][0], 
                          reverse=True)[:beam_width])
    
    # Return best sequence
    best_seq = max(beam.items(), key=lambda x: x[1][0])
    return list(best_seq[0]), best_seq[1][0]


# Example usage with your MFCC data
# Assuming you have log_probs from your model

# Character mapping
chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
blank_id = 26

# Get log probabilities from your model
log_probs_np = log_probs[0].detach().cpu().numpy()  # (32, 27)

# Decode with beam search
best_path, score = ctc_beam_search(
    log_probs_np, 
    beam_width=10, 
    blank_id=blank_id
)

# Convert to text
decoded_text = ''.join([idx_to_char[i] for i in best_path])
print(f"Decoded: '{decoded_text}'")
print(f"Log probability: {score:.4f}")
```

---

## Summary

**Beam search** is a heuristic search algorithm that:
- Explores multiple paths simultaneously
- Prunes to keep only the most promising K paths
- Balances between greedy (K=1) and exhaustive search (K=∞)
- Provides better decoding accuracy than greedy while being computationally feasible

For CTC specifically, it respects the blank token and character repetition rules while finding the most likely decoded sequence.