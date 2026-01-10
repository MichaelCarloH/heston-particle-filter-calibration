"""
Parameter estimation example.

This example demonstrates:
1. Estimating parameters using PMMH
2. Using estimated parameters for filtering
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

# Estimate parameters using PMMH
print("\nEstimating parameters using PMMH (this will take several minutes)...")
estimation_results = hest.estimate_parameters(
    method='pmmh',
    n_iter=10000,
    n_particles=200,
    verbose=True
)

# Print estimated parameters
print("\n=== Estimated Parameters ===")
for param, value in hest.estimated_params.items():
    print(f"{param:8s}: {value:.4f}")

# Re-run filter with estimated parameters
print("\nRe-running particle filter with estimated parameters...")
hest.fit(n_particles=2000, verbose=True)

print(f"\nLog-likelihood with estimated parameters: {hest.log_likelihood:.2f}")

