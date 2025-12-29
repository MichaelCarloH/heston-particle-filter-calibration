# Heston Particle Filter Calibration

This project implements and analyzes Bayesian filtering techniques for estimating the latent volatility and parameters of the Heston stochastic volatility model.

## Features

- **Heston State-Space Model**: Discrete-time implementation with time-varying risk-free rates
- **Bootstrap Particle Filter**: Using Chopin's `particles` library
- **Parameter Estimation**: PMMH (Particle Marginal Metropolis-Hastings) and Simplex methods
- **QMC Analysis**: Quasi-Monte Carlo variance reduction analysis
- **Data Loading**: Automatic download of S&P 500 and risk-free rate data
- **Visualization**: Volatility comparison with VIX and realized volatility

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from heston import HestonModel

# Initialize model
hest = HestonModel(dt=1/252)

# Load S&P 500 data
hest.load_data(ticker="^GSPC", start="2007-01-01", end="2025-12-31")

# Set parameters
hest.kappa = 2.5
hest.theta = 0.05
hest.sigma = 0.78
hest.rho = -0.6

# Run particle filter
hest.fit(n_particles=2000)
print(f"Log-likelihood: {hest.log_likelihood:.2f}")
```

### QMC Analysis

```python
# Compare QMC vs standard Monte Carlo
qmc_results = hest.qmc(n_runs=30, n_particles=2000)
print(f"Variance reduction: {qmc_results['variance_reduction_pct']:.2f}%")
```

### Parameter Estimation

```python
# Estimate parameters using PMMH
hest.estimate_parameters(method='pmmh', n_iter=10000, n_particles=200)

# Re-run filter with estimated parameters
hest.fit(n_particles=2000)
```

## Project Structure

```
heston-particle-filter-calibration/
├── heston/                  # Main package
│   ├── __init__.py         # Package exports
│   ├── model.py            # HestonModel and HestonSSM classes
│   ├── utils.py            # Data loading utilities
│   ├── qmc.py              # QMC analysis
│   └── estimation.py       # Parameter estimation methods
├── examples/               # Example scripts
│   ├── basic_usage.py
│   ├── qmc_analysis.py
│   └── parameter_estimation.py
├── notebooks/              # Jupyter notebooks (original exploration)
│   └── exploration/
│       └── heston_ssm_bootstrap_pf.ipynb
└── tests/                  # Unit tests
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py`: Simple particle filtering
- `qmc_analysis.py`: QMC variance reduction analysis
- `parameter_estimation.py`: Parameter estimation with PMMH

## API Reference

### HestonModel

Main class for Heston model analysis.

#### Methods

- `load_data(ticker, start, end, risk_free_ticker)`: Load market data
- `fit(n_particles, resampling, collect_moments, verbose)`: Run particle filter
- `qmc(n_runs, n_particles, qmc_type, compare_standard, verbose)`: QMC analysis
- `estimate_parameters(method, n_iter, n_particles, prior, **kwargs)`: Estimate parameters
- `get_results()`: Get comprehensive results dictionary

## Requirements

- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy
- yfinance
- particles (Chopin's particle filtering library)

## Project Status

✅ **Completed:**
- Heston SSM implementation
- Bootstrap particle filter
- PMMH parameter estimation
- QMC analysis framework
- Data loading utilities
- Modular class structure

🚧 **In Progress:**
- Extended Kalman Filter comparison
- EM algorithm implementation
- Guided filter implementation

## References

- Chopin, N., & Papaspiliopoulos, O. (2020). *An introduction to sequential Monte Carlo methods*. Springer.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *The Review of Financial Studies*, 6(2), 327-343.

## License

MIT
