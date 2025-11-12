### 🎯 The Goal

We want to model a set of 1D data points using a **Gaussian Mixture Model (GMM)** with $K=2$ components. We will run **one full iteration** of the Expectation-Maximization (EM) algorithm to find better parameters.

---

### ## Step 0: Setup & Initial Guess

- Data ($N=5$ points):
    
    $X = \{1, 2, 5, 8, 9\}$
    
- **Components ($K=2$):** Component 1 and Component 2.
    
- **Initial Guesses:** We have to start with a guess for our parameters.
    
    - **Priors:** Let's guess they are equally likely.
        
        - $\pi_1 = 0.5$
            
        - $\pi_2 = 0.5$
            
    - **Means:** Let's guess two means that are reasonably close to the two clusters we see.
        
        - $\mu_1 = 2.0$
            
        - $\mu_2 = 8.0$
            
    - **Variance (fixed):** To keep the math simple, we'll assume a fixed variance for both.
        
        - $\sigma_1^2 = 2.0$
            
        - $\sigma_2^2 = 2.0$
            

---

### ## Step 1: E-Step (Calculate Responsibilities)

The **E-Step** answers the question: "How responsible is each component ($k$) for each data point ($n$)?" We calculate this using the `r_nk` formula from your slides.

> **Lecture Formula:**
> 
> $$r_{nk} = \frac{\pi_k \,\mathcal{N}(x_n\mid\mu_k,\Sigma_k)}{\sum_{j=1}^K \pi_j \,\mathcal{N}(x_n\mid\mu_j,\Sigma_j)}$$

First, we calculate the likelihood $\mathcal{N}(x_n \mid \mu_k, \sigma_k^2)$ of each point under each component.

The 1D Gaussian formula is $\mathcal{N}(x) \propto \exp\left(-\frac{(x-\mu)^{2}}{2\sigma^{2}}\right)$.

With our parameters, this is $\exp\left(-\frac{(x_n-\mu_k)^{2}}{4}\right)$.

| Data Point ($x_n$) | Likelihood (Comp 1) $\mathcal{N}(x_n | \mu_1=2, \sigma^2=2)$ | Likelihood (Comp 2) $\mathcal{N}(x_n | \mu_2=8, \sigma^2=2)$ |

| :--- | :--- | :--- |

| $x_1=1$ | $\exp(-\frac{(1-2)^2}{4}) = \exp(-0.25) \approx 0.779$ | $\exp(-\frac{(1-8)^2}{4}) = \exp(-12.25) \approx 0.000$ |

| $x_2=2$ | $\exp(-\frac{(2-2)^2}{4}) = \exp(0) = 1.000$ | $\exp(-\frac{(2-8)^2}{4}) = \exp(-9) \approx 0.000$ |

| $x_3=5$ | $\exp(-\frac{(5-2)^2}{4}) = \exp(-2.25) \approx 0.105$ | $\exp(-\frac{(5-8)^2}{4}) = \exp(-2.25) \approx 0.105$ |

| $x_4=8$ | $\exp(-\frac{(8-2)^2}{4}) = \exp(-9) \approx 0.000$ | $\exp(-\frac{(8-8)^2}{4}) = \exp(0) = 1.000$ |

| $x_5=9$ | $\exp(-\frac{(9-2)^2}{4}) = \exp(-12.25) \approx 0.000$ | $\exp(-\frac{(9-8)^2}{4}) = \exp(-0.25) \approx 0.779$ |

Now we calculate the responsibilities ($r_{nk}$). Since we guessed $\pi_1 = \pi_2 = 0.5$, the $\pi$ terms in the formula cancel out, and the responsibility is just the likelihood of one component divided by the total likelihood.

- **For $x_3=5$:** $r_{3,1} = \frac{0.105}{0.105 + 0.105} = \mathbf{0.5}$
    
- **For $x_1=1$:** $r_{1,1} = \frac{0.779}{0.779 + 0.000} \approx \mathbf{1.0}$
    

Here are the final responsibilities for all points:

|**Data Point (xn​)**|**Resp. rn1​ (Comp 1)**|**Resp. rn2​ (Comp 2)**|
|---|---|---|
|**$x_1=1$**|1.000|0.000|
|**$x_2=2$**|1.000|0.000|
|**$x_3=5$**|**0.500**|**0.500**|
|**$x_4=8$**|0.000|1.000|
|**$x_5=9$**|0.000|1.000|

**E-Step Takeaway:** The points $1$ and $2$ belong to Component 1. The points $8$ and $9$ belong to Component 2. The middle point $5$ is **equally (50/50) responsible** by both components.

---

### ## Step 2: M-Step (Re-Estimate Parameters)

The **M-Step** uses these "soft" responsibilities to compute new, better parameters.

**1. Calculate $N_k$ (Effective number of points for each component):**

> $N_k = \sum_{n=1}^N r_{nk}$

- $N_1 = 1.0 + 1.0 + 0.5 + 0.0 + 0.0 = \mathbf{2.5}$
    
- $N_2 = 0.0 + 0.0 + 0.5 + 1.0 + 1.0 = \mathbf{2.5}$
    
    (This makes sense: each component is responsible for 2.5 of the 5 data points).
    

**2. Update Priors $\pi_k$:**

> $\pi_k = \frac{N_k}{N}$

- $\pi_1^{\text{new}} = 2.5 / 5 = \mathbf{0.5}$
    
- $\pi_2^{\text{new}} = 2.5 / 5 = \mathbf{0.5}$
    
    (The priors didn't change because the data was symmetric).
    

3. Update Means $\mu_k$:

This is a "weighted average," using the responsibilities as weights.

> $\mu_k = \frac{1}{N_k} \sum_{n=1}^N r_{nk} x_n$

- **New Mean $\mu_1$:**
    
    $$ \mu_1^{\text{new}} = \frac{1}{2.5} \left[ (1.0 \times 1) + (1.0 \times 2) + (0.5 \times 5) + (0.0 \times 8) + (0.0 \times 9) \right]$$
    
    $$ \mu_1^{\text{new}} = \frac{1}{2.5} \left[ 1 + 2 + 2.5 \right] = \frac{5.5}{2.5} = \mathbf{2.2}$$
    
- **New Mean $\mu_2$:**
    
    $$ \mu_2^{\text{new}} = \frac{1}{2.5} \left[ (0.0 \times 1) + (0.0 \times 2) + (0.5 \times 5) + (1.0 \times 8) + (1.0 \times 9) \right]$$
    
    $$ \mu_2^{\text{new}} = \frac{1}{2.5} \left[ 2.5 + 8 + 9 \right] = \frac{19.5}{2.5} = \mathbf{7.8}$$
    

---

### ## Conclusion of Iteration 1

We have completed one loop. Look at how the parameters have "learned" from the data:

- **Mean 1:** Started at $\mu_1 = 2.0 \rightarrow$ **Moved to $\mu_1^{\text{new}} = 2.2$**
    
- **Mean 2:** Started at $\mu_2 = 8.0 \rightarrow$ **Moved to $\mu_2^{\text{new}} = 7.8$**
    

The mean $\mu_1$ was "pulled" slightly up, and the mean $\mu_2$ was "pulled" slightly down. This is because the ambiguous middle point $x=5$ exerted a "pull" on both components, weighted by its 50% responsibility.

The algorithm would now take these new parameters and repeat the E-Step and M-Step until the parameters stop changing.