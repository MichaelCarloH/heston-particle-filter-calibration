"""
QMC (Quasi-Monte Carlo) analysis example.

This example demonstrates:
1. Running QMC analysis to compare variance reduction
2. Statistical testing of variance reduction
"""

from heston import HestonModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize model
hest = HestonModel(dt=1/252)

# Load data
print("Loading market data...")
hest.load_data(ticker="^GSPC", start="2007-01-01", end="2025-12-31")

# Set parameters
hest.kappa = 2.5
hest.theta = 0.05
hest.sigma = 0.78
hest.rho = -0.6

# Run QMC analysis
print("\nRunning QMC analysis (this may take a while)...")
qmc_results = hest.qmc(n_runs=30, n_particles=2000, verbose=True)

# Print summary
print("\n=== QMC Analysis Summary ===")
if 'variance_reduction_pct' in qmc_results:
    print(f"Variance reduction: {qmc_results['variance_reduction_pct']:.2f}%")
    print(f"t-test p-value: {qmc_results['t_test']['pvalue']:.4f}")
    print(f"F-test p-value: {qmc_results['f_test']['pvalue']:.4f}")
    print(f"Significant variance reduction: {qmc_results['f_test']['significant']}")

