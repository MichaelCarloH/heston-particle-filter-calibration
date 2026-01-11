# Heston Stochastic Volatility Model: Particle Filter Calibration

**Project for Sequential Monte Carlo Methods Course**

This project implements Bayesian filtering techniques for estimating latent volatility and parameters of the Heston stochastic volatility model using S&P 500 data.

---

## 📁 Project Structure

```
heston-particle-filter-calibration/
├── README.md
├── requirements.txt
│
├── heston/                            # Main package
│   ├── __init__.py
│   ├── model.py                       # HestonModel and HestonSSM classes
│   ├── estimation.py                  # PMMH, Simplex, EM implementations
│   ├── qmc.py                         # QMC variance analysis
│   └── utils.py                       # Data loading utilities
│
├── notebooks/
│   ├── exploration/                   # Main project notebooks
│   │   ├── step1_bootstrap_qmc.ipynb
│   │   ├── step2_parameter_estimation_comparison.ipynb
│   │   └── step3_guided_ekf_comparison.ipynb
│   │
│   └── tutorials/                     # Reference tutorials
│
├── examples/                          # Example scripts
└── tests/                             # Unit tests
```

---

## 📓 Notebooks

### Step 1: Bootstrap Filter + QMC Analysis
**File**: `notebooks/exploration/step1_bootstrap_qmc.ipynb`

**What it does**:
- Implements bootstrap particle filter using Chopin's `particles` library
- Approximates log-likelihood for **fixed parameter vector** $\theta$
- Estimates hidden volatility path from S&P 500 returns
- Analyzes whether QMC (SQMC) reduces variance of log-likelihood estimates
- Compares filtered volatility with realized volatility and VIX

**Key components**:
- Heston model discretization and state-space formulation
- Bootstrap filter with 2000 particles
- QMC variance analysis (30 runs, statistical tests)
- Volatility validation against market measures

---

### Step 2: Parameter Estimation Methods Comparison
**File**: `notebooks/exploration/step2_parameter_estimation_comparison.ipynb`

**What it does**:
- Explains paper's parameter estimation method (NMLE with switching filters)
- Compares three MLE methods:
  - **PMMH** (Particle Marginal Metropolis-Hastings): Bayesian MCMC
  - **Simplex** (Nelder-Mead): Direct optimization
  - **EM** (Expectation-Maximization): Iterative optimization
- Estimates parameters from S&P 500 data
- Compares results and discusses trade-offs

---

### Step 3: Guided Filter + Extended Kalman Filter
**File**: `notebooks/exploration/step3_guided_ekf_comparison.ipynb`

**What it does**:
- Implements **guided particle filter** (optimal proposal)
- Implements **Extended Kalman Filter (EKF)**
- Compares bootstrap filter, guided filter, and EKF on:
  - Log-likelihood estimates
  - Filtered volatility paths
  - Effective Sample Size (ESS)
  - Computation time
- Analyzes when each method is most appropriate

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from heston import HestonModel

# Initialize and load data
hest = HestonModel(dt=1/252)
hest.load_data(ticker="^GSPC", start="2007-01-01", end="2025-12-31")

# Set parameters and run filter
hest.kappa = 2.5
hest.theta = 0.05
hest.sigma = 0.78
hest.rho = -0.6

hest.fit(n_particles=2000)
print(f"Log-likelihood: {hest.log_likelihood:.2f}")
```

---

## 📚 Dependencies

- Python 3.7+
- `numpy`, `pandas`, `matplotlib`, `scipy`
- `yfinance` - Market data download
- `particles` - Chopin's particle filtering library

---

## 📖 References

1. **Chopin, N., & Papaspiliopoulos, O.** (2020). *An introduction to sequential Monte Carlo methods*. Springer.
2. **Heston, S. L.** (1993). A closed-form solution for options with stochastic volatility. *The Review of Financial Studies*, 6(2), 327-343.
3. **Chopin's particles library**: https://github.com/nchopin/particles
