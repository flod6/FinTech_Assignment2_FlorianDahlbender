#----------------------------------
# 1. Set Up
#----------------------------------


# Load Packages
import pandas as pd
import numpy as np
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import pickle


# Import Custiom Objects and Functions
from Model_Development.utils.utils_data_prep import (generate_data, 
                                   create_graph, 
                                   graph_sanity_check,
                                   plot_graph,
                                   extract_all_accounts)

# Set paths
dir = Path(__file__).resolve().parent.parent


# Set Random Seed
np.random.seed(187)

#----------------------------------
# 2. Simulate Data
#----------------------------------


# Define Number of Accounts and Transactions
n_accounts = 5000
n_transactions = 25000

# Define Number of Layering Chains and Length of Each Chain
n_chains = 15
chain_length = 5

# Define Number of Accounts to Simulate Smurfing
n_smurfing = 75
num_smurfing_deposits = 5

# Create DataFrame with Simulated Transactions
df = generate_data(n_accounts, n_transactions, n_chains, chain_length, n_smurfing, num_smurfing_deposits)

# Store the DataFrame to a CSV file
df.to_csv(dir / "Data" / "Processed" / "simulated_transactions.csv", index=False)

# Extract all accounts with positive labels / i.e. accounts that are part of a layering chain or smurfing
all_accounts = extract_all_accounts(df)

# Retrive number of positive cases
n_positive_cases = all_accounts[all_accounts["is_suspicious"] == 1].shape[0]

# Store all accounts to a CSV file
all_accounts.to_csv(dir / "Data" / "Processed" / "accounts_train.csv", index=False)

#----------------------------------
# 3. Create Graph
#----------------------------------


# Create the graph from the DataFrame
graph = create_graph(df)

# Create Sanity check for the graph
graph_sanity_check(graph, df)

# Plot the graph
plot_graph(graph, df, limit_nodes=250)

# Store the graph to a file
with open(dir / "Data" / "Processed" / "simulated_graph.gpickle", "wb") as f:
    pickle.dump(graph, f)




