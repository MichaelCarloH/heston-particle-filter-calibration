"""
Basic usage example for HestonModel.

This example demonstrates:
1. Loading market data
2. Running particle filter
3. Viewing results
"""

from heston import HestonModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize model
hest = HestonModel(dt=1/252)

# Load S&P 500 data
print("Loading market data...")
hest.load_data(ticker="^GSPC", start="2007-01-01", end="2025-12-31")

# Set initial parameters (or estimate them later)
hest.kappa = 2.5
hest.theta = 0.05
hest.sigma = 0.78
hest.rho = -0.6

# Run bootstrap particle filter
print("\nRunning particle filter...")
results = hest.fit(n_particles=2000, verbose=True)

# Print results
print(f"\n=== Results ===")
print(f"Log-likelihood: {hest.log_likelihood:.2f}")
print(f"Filtered volatility (mean): {hest.filtered_volatility.mean():.4f}")
print(f"Filtered volatility (std): {hest.filtered_volatility.std():.4f}")

# Get comprehensive results
all_results = hest.get_results()
print(f"\nParameters: {all_results['parameters']}")

