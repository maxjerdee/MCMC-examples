"""
Inferring hierarchy in directed networks
"""

import pandas as pd
import numpy as np
import stan
import matplotlib.pyplot as plt
import os

data_folder = "data"

# Define the model
STAN_model_code = """
data {
  int<lower = 0> N;                             // Number of players
  int<lower = 0> M;                             // Number of unique pairings
  array[M] int<lower=1, upper = N> player1;     // The index of "player1" in each pairing
  array[M] int<lower=1, upper = N> player2;     // The index of "player2" in each pairing
  array[M] real<lower = 0> A12;                 // Number of wins recorded by player1 against player2
}
parameters {
  vector[N] scores;                             // Score of each player
  real<lower = 0> sigma;                         // Depth of competition
}
model {
  target += log(1/(4*(1 + sigma^2/16)));         // P(sigma), cauchy prior of width 4 (uniform over theta)
  for( k in 1:N){                               // P(s_i), prior on scores
    target += -(scores[k])^2;                   // Prior has variance 1/2 so that the difference between two scores is a Gaussian of width 1
  }
  for(i in 1:M){                                // P(A|s_i,sigma), model likelihood
    real ds = scores[player1[i]] - scores[player2[i]];  // Score difference
    target += A12[i]*log(1/(1+e()^(-sigma*ds))); // Probability of times that player1 beat player2
  }
}
"""

dataset_samples = {}

def fit_to_data(filename):
    # Load data
    string_indices_dict = {}  # Dictionary of {string: index} for quick access

    adj_matrix = np.loadtxt(f"{data_folder}/{filename}", dtype=int)
    n = len(adj_matrix)

    match_list = []
    for i in range(n):
        for j in range(n):
            # Give each player a label from player_0 to player_(n-1) depending on where they appear in the given matrix
            player_i_label = f"player_{i}"
            player_j_label = f"player_{j}"
            for t in range(adj_matrix[i, j]):  # Add the number of matches which occured
                match_list.append({"winner": player_i_label, "loser": player_j_label})

    for match in match_list:
        # Find (or assign) indices to the new labels
        winner_label = match["winner"]
        loser_label = match["loser"]
        if winner_label not in string_indices_dict.keys():
            string_indices_dict.update({winner_label: len(string_indices_dict)})
        if loser_label not in string_indices_dict.keys():
            string_indices_dict.update({loser_label: len(string_indices_dict)})

    n = len(string_indices_dict)  # Number of players

    # Initializing STAN_data variables to be built up
    m = 0  # Number of unique pairings
    player1 = []  # The index of "player1" in each pairing
    player2 = []  # The index of "player2" in each pairing
    A12 = []  # Number of wins recorded by player1 against player2

    # Map the pairings present in the data to an index in the STAN_data arrays
    pairing_indices_dict = {}

    for match in match_list:
        winner_index = string_indices_dict[match["winner"]]
        loser_index = string_indices_dict[match["loser"]]

        # Store a pairing as the tuple [hashable!] (winner_index,loser_index)
        pairing = (winner_index, loser_index)

        # Assign this pairing an index if haven't already and add entries to
        # the STAN_data variables to represent this pairing.
        if pairing not in pairing_indices_dict.keys():
            pairing_indices_dict[pairing] = len(pairing_indices_dict)
            # Participants
            player1.append(pairing[0] + 1)  # STAN indices are unary
            player2.append(pairing[1] + 1)  # STAN indices are unary
            # Win counts
            A12.append(0)
            # Number of edges
            m += 1

        # Increment the STAN_data variables
        A12[pairing_indices_dict[pairing]] += 1

    STAN_data = {
        "N": int(n),
        "M": int(m),  # Number of unique pairings
        "player1": player1,  # The index of "player1" in each pairing
        "player2": player2,  # The index of "player2" in each pairing
        "A12": A12,
    }  # Number of wins recorded by player1 against player2

    # Build the model
    STAN_model = stan.build(STAN_model_code, data=STAN_data)

    # Fit the data
    fit = STAN_model.sample(num_chains=4, num_samples=1000)
    samples = fit.to_frame()

    # Print summary statistics
    print(f"Results for {filename}:")
    print("=" * 40)
    print(f"Depth (sigma): {samples['sigma'].mean():.3f} ± {samples['sigma'].std():.3f}")

    # Save the dataset samples so we can plot them all together
    dataset_samples[filename] = samples

dataset_filenames = ["basketball.txt", "business_depts.txt", "baboons.txt"]

for dataset_filename in dataset_filenames:
    fit_to_data(dataset_filename)

# Plot the histogram of the depth posteriors
plt.figure(figsize=(6, 4))
colors = ['r','g','b']
for (filename, samples), color in zip(dataset_samples.items(), colors):
    plt.hist(samples['sigma'], bins=30, density=True, alpha=0.5, label=filename[:-4], color=color)
    plt.axvline(x=samples['sigma'].mean(), color=color, linestyle='--')
plt.title('Posterior Distribution of Depth (sigma)')
plt.xlabel('Depth (sigma)')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()
plt.savefig('network_hierarchy_results.png')
print("Plot saved as 'network_hierarchy_results.png'")
