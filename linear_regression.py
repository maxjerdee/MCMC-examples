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

# Load data
data_df = pd.read_csv(os.path.join(data_folder, "linear_regression_data.csv"))

# Prepare data for STAN (dictionary that matches the STAN code)
STAN_data = {
    'N': len(data_df),
    'x': data_df['x'].values,
    'y': data_df['y'].values
}

# Define the STAN model
STAN_model_code = """
data {
  int<lower=0> N;      // number of data points
  vector[N] x;         // predictor
  vector[N] y;         // outcome
}

parameters {
  real b;          // intercept
  real m;           // slope
  real<lower=0> sigma; // error sd
}

model {
  // priors
  b ~ normal(0, 10);
  m ~ normal(0, 10);
  sigma ~ exponential(1);
  
  // likelihood
  y ~ normal(b + m * x, sigma);
}
"""

# Build the model (note: the first time you run this, STAN will take some time here, later runs will cache the compilation)
STAN_model = stan.build(STAN_model_code, data=STAN_data)

# Fit the data 
fit = STAN_model.sample(num_chains=4, num_samples=1000)
samples = fit.to_frame()

# Print summary statistics
print("Linear Regression Results:")
print("=" * 40)
print(f"Intercept (b): {samples['b'].mean():.3f} ± {samples['b'].std():.3f}")
print(f"Slope (m): {samples['m'].mean():.3f} ± {samples['m'].std():.3f}")
print(f"Sigma: {samples['sigma'].mean():.3f} ± {samples['sigma'].std():.3f}")

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))

# Plot 1: fit to the data
# Plot the original data
axes[0].scatter(data_df['x'], data_df['y'], alpha=0.7, label='Observed data')

# Plot the fit using the single expected values
axes[0].plot(data_df['x'], samples['b'].mean() + samples['m'].mean() * data_df['x'], 
             'b-', alpha=0.5, label='EAP parameters')

# Plot sampled fits from the posterior
for i in range(0, len(samples), 50):  # Plot every 50th sample
    b_sample = samples.iloc[i]['b']
    m_sample = samples.iloc[i]['m']
    if i == 0:
        axes[0].plot(data_df['x'], b_sample + m_sample * data_df['x'], 
                    'r-', alpha=0.1, label='Posterior samples')
    else:
        axes[0].plot(data_df['x'], b_sample + m_sample * data_df['x'], 
                    'r-', alpha=0.1)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Data and fits')
    axes[0].legend()

# Plot 2: Joint posterior distribution of the slope and intercept
axes[1].scatter(samples['b'], samples['m'], alpha=0.5)
axes[1].set_xlabel('Intercept (b)')
axes[1].set_ylabel('Slope (m)')
axes[1].set_title('Joint Posterior Distribution of b and m')

    
plt.tight_layout()
plt.savefig('linear_regression_results.png')
print("Plot saved as 'linear_regression_results.png'")
    