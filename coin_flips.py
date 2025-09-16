"""
Simple Linear Regression using PyStan

This example demonstrates Bayesian linear regression using STAN.
"""

import pandas as pd
import numpy as np
import stan
import matplotlib.pyplot as plt
import os

data_folder = "data"

data_filename = "coin_flips.txt"

# Load data
outcomes = np.loadtxt(f"{data_folder}/{data_filename}", dtype=str)
# Convert to binary
outcomes = np.where(outcomes == 'heads', 1, 0)

# Prepare data for STAN (dictionary that matches the STAN code)
stan_data = {
    'N': len(outcomes),
    'outcomes': outcomes
}

# Define the STAN model
stan_model_code = """
data {
  int<lower=0> N;      // number of data points
  array[N] int<lower=0, upper = 1> outcomes;         // outcomes
}

parameters {
  real p;          // probability of heads
}

model {
  // priors
  p ~ beta(20, 20); // Peaked beta distribution

  // likelihood (independent flips)
  for (n in 1:N) {
    outcomes[n] ~ bernoulli(p);
  }
}
"""

# Build the model (note: the first time you run this, STAN will take some time here, later runs will cache the compilation)
stan_model = stan.build(stan_model_code, data=stan_data)

# Fit the data 
fit = stan_model.sample(num_chains=4, num_samples=1000)
samples = fit.to_frame()

# Print summary statistics
print("Bayesian Logistic Regression Results:")
print("=" * 40)
print(f"Probability of heads (p): {samples['p'].mean():.3f} ± {samples['p'].std():.3f}")

# Plot results
plt.figure(figsize=(4, 4))
plt.hist(samples['p'], bins=30, density=True, alpha=0.7)
plt.xlim(0, 1)
plt.axvline(x=samples['p'].mean(), color='r', linestyle='--', label='Mean')
plt.title('Posterior distribution of heads probability')
plt.xlabel('Prob. of heads (p)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('coin_flips_results.png')
print("Plot saved as 'coin_flips_results.png'")
