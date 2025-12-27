# Heston SSM Bootstrap Particle Filter - Project Summary

## What Has Been Completed

### 1. Heston State-Space Model Implementation ✓
- **Discrete-time Heston model** implemented as a `StateSpaceModel` class
- **State transition**: Euler discretization of CIR process for variance dynamics
- **Observation model**: Log-returns following normal distribution conditional on variance
- **Time-varying risk-free rate**: Integrated 3-month Treasury rate (^IRX) from Yahoo Finance
- Model handles both constant and time-varying risk-free rates

### 2. Bootstrap Particle Filter ✓
- Implemented using Chopin's `particles` library
- **Synthetic data testing**: Verified filter can reconstruct hidden volatility paths
- **S&P 500 application**: Applied to real market data (2007-2025, ~4,769 observations)
- **Particle count**: 2,000 particles with systematic resampling
- **Results**: 
  - Estimated log-likelihood: ~15,385 (for fixed parameters)
  - Filtered volatility shows correlation of 0.90 with VIX
  - Correlation of 0.94 with 21-day realized volatility

### 3. Parameter Estimation (PMMH) ✓
- **Particle Marginal Metropolis-Hastings** implemented
- **Prior distributions**:
  - κ (kappa): Gamma(2.0, 1.0)
  - θ (theta): Gamma(2.0, 40.0)
  - σ (sigma): Gamma(2.0, 2.5)
  - ρ (rho): Uniform(-0.99, 0.99)
  - v₀: Fixed at 0.04
  - r: Time-varying (observed data, not estimated)
- **MCMC chain**: 10,000 iterations with 200 particles per iteration
- **Posterior estimates**:
  - κ: 2.24
  - θ: 0.035
  - σ: 0.46
  - ρ: 0.06
- **Calibrated model**: Re-run filter with estimated parameters shows improved log-likelihood (~15,460)

### 4. Volatility Comparison ✓
- **Three volatility measures compared**:
  1. PF-Heston Volatility (instantaneous, physical measure)
  2. Realized Volatility (21-day rolling window)
  3. VIX Implied Volatility (forward-looking, risk-neutral measure)
- **Results**: Strong correlations between all three measures, validating the model

### 5. Data Handling ✓
- **S&P 500 data**: Downloaded and processed (2007-2025)
- **Risk-free rate**: Time series from 3-month Treasury Bill (^IRX)
- **Data preprocessing**: Log-returns computed, missing data handled

---

## What Remains to Be Done

Based on Professor Chopin's email requirements:

### 1. QMC (Quasi-Monte Carlo) Analysis ⚠️ **NOT STARTED**
**Requirement**: "you can also try to see whether QMC helps here, i.e. does it reduce the variance of the estimates on repeated runs"

**What to do**:
- Run the bootstrap filter multiple times (e.g., 30-50 runs) with:
  - Standard Monte Carlo (`qmc=False`)
  - Quasi-Monte Carlo (`qmc={'smc': False, 'sqmc': True}`)
- Compare variance of log-likelihood estimates between the two methods
- Use `particles.multiSMC()` function for parallel runs
- Report whether QMC provides variance reduction and quantify the improvement

**Expected outcome**: Box plots or variance statistics showing QMC vs standard MC performance

---

### 2. Parameter Estimation Methods Discussion ⚠️ **NOT STARTED**
**Requirement**: "explain how the authors estimate the theta based on real data, and compare to other (MLE) methods, see the corresponding chapter. In particular, discuss you could use the simplex and / or the EM algorithm."

**What to do**:
- **Read the paper** (adaptive Heston paper) to understand:
  - How authors estimate parameters
  - What MLE methods they use
  - Whether they mention simplex or EM algorithm
- **Implement alternative methods**:
  - **Simplex method** (Nelder-Mead): Direct optimization of log-likelihood
  - **EM algorithm**: For state-space models, this involves:
    - E-step: Run particle filter/smoother
    - M-step: Update parameters using sufficient statistics
- **Compare methods**:
  - PMMH (already done)
  - Simplex optimization
  - EM algorithm
  - Any method from the paper
- **Discussion**: Compare convergence, computational cost, accuracy

**Expected outcome**: 
- Written explanation of methods
- Code implementations
- Comparison table/results

---

### 3. Extended Kalman Filter (EKF) Comparison ⚠️ **OPTIONAL - NOT STARTED**
**Requirement**: "If time permits, you can either try to replicate their comparison between the bootstrap filter and the extended Kalman filter"

**What to do**:
- **Implement Extended Kalman Filter** for Heston model:
  - Linearize the observation equation around current state estimate
  - Implement EKF prediction and update steps
- **Compare with bootstrap filter**:
  - Log-likelihood estimates
  - Filtered volatility paths
  - Computational time
  - Accuracy (if true volatility known from simulation)
- **Resources**: Online tutorials on EKF for stochastic volatility models

**Expected outcome**: 
- EKF implementation
- Side-by-side comparison with bootstrap filter
- Discussion of pros/cons

---

### 4. Guided Filter Implementation ⚠️ **OPTIONAL - NOT STARTED**
**Requirement**: "If time permits, you can either try to implement a guided filter"

**What to do**:
- **Implement optimal proposal** for Heston model:
  - The optimal proposal is `p(X_t | X_{t-1}, Y_t)`
  - For Heston: This involves combining transition and observation densities
  - May require numerical methods or approximations
- **Compare guided vs bootstrap filter**:
  - Effective sample size (ESS)
  - Variance of log-likelihood estimates
  - Computational cost
- **Note**: The `particles` library supports guided filters via `ssm.GuidedPF` class
- Need to implement `proposal()` and `proposal0()` methods in `HestonSSM`

**Expected outcome**:
- Guided filter implementation
- Performance comparison with bootstrap filter
- Analysis of variance reduction

---

## Technical Notes

### Current Implementation Status
- ✅ Heston SSM class fully functional
- ✅ Bootstrap filter working on real data
- ✅ PMMH parameter estimation complete
- ✅ Volatility filtering and comparison done
- ⚠️ QMC analysis: Not started
- ⚠️ Alternative estimation methods: Not started
- ⚠️ EKF: Not started
- ⚠️ Guided filter: Not started

### Key Files
- `notebooks/exploration/heston_ssm_bootstrap_pf.ipynb`: Main implementation notebook
- `main.py`: Currently just a placeholder

### Dependencies
- `particles` library (Chopin's package)
- `numpy`, `matplotlib`, `pandas`
- `yfinance` for market data

---

## Recommendations for Completion

### Priority 1 (Required):
1. **QMC Analysis** - Should be straightforward using `multiSMC()` function
2. **Parameter Estimation Discussion** - Read paper and implement at least one alternative method (simplex or EM)

### Priority 2 (If Time Permits):
3. **EKF Comparison** - Requires implementing EKF from scratch
4. **Guided Filter** - Requires deriving optimal proposal distribution

### Timeline Suggestion:
- **Week 1**: QMC analysis + Read paper on parameter estimation
- **Week 2**: Implement simplex/EM + Write comparison
- **Week 3**: EKF or Guided filter (choose one)
- **Week 4**: Finalize slides and prepare presentation

---

## Questions to Address in Final Report

1. **Does QMC reduce variance?** By how much?
2. **How do different parameter estimation methods compare?**
   - PMMH vs Simplex vs EM
   - Convergence properties
   - Computational efficiency
3. **How does bootstrap filter compare to EKF?** (if implemented)
4. **Does guided filter improve performance?** (if implemented)
5. **What are the limitations of the current approach?**
6. **How do results compare to the paper?**

---

## Deliverables (Due: Sunday 11th January)

- ✅ Code (GitHub repository)
- ⚠️ Slides (15-minute presentation)
- ⚠️ Jupyter notebook with plots (optional, for viva)

**Important**: Show numerical common sense - run algorithms multiple times and report variability of results.

