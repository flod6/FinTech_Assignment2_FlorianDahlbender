#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import torch
import torch_geometric as pyg
from sklearn.metrics import classification_report
from sklearn.utils import resample

# Import Custom Objects and Functions
from Model_Development.utils.utils_data_prep import generate_data, create_graph, extract_all_accounts, plot_graph
from Model_Development.utils.utils_train_GNN import create_pyg_graph, extract_node_embeddings, corruption, GraphSage
from Model_Development.utils.utils_train_online_MAB import reward_function, setup_mab_model


# Set paths
dir = Path(__file__).resolve().parent.parent


# Load Models
with open(dir / "Model_Development" / "Models" / "mab_model.pkl", "rb") as f:
    mab = pickle.load(f)


# Set Random Seed
np.random.seed(187)
torch.manual_seed(6)


#----------------------------------
# 1. Create Test Data
#----------------------------------

# Set Parameters
n_accounts = 5000
n_transactions = 25000

# Simulate Data with different parameters
n_chains = 5
chain_length = 5
n_smurfing = 5
num_smurfing_deposits = 10

# Simulate the Data
df = generate_data(n_accounts, n_transactions, n_chains, chain_length, n_smurfing, num_smurfing_deposits)

# Extract all accounts with their lables
all_accounts = extract_all_accounts(df)

# Create Graph from testing data
graph = create_graph(df)

# Plot the graph
plot_graph(graph, df, 1000)


#----------------------------------
# 2. Create Test Embeddings
#----------------------------------

# Create PyTorch Geometric graph
graph_pyg = create_pyg_graph(graph)


# Set up the GNN model
encoder = GraphSage(
    in_channels=graph_pyg.num_node_features,
    hidden_channels=64,
    num_layers=5,
    out_channels=4,
    dropout=0,
    aggregator="mean"
)
model = pyg.nn.DeepGraphInfomax(
    hidden_channels=4,
    encoder=encoder,
    summary=lambda z, *args, **kwargs: torch.sigmoid(torch.mean(z, dim=0)),
    corruption=corruption
)

# Load the model state
model.load_state_dict(torch.load(dir / "Model_Development" / "Models" / "gnn_model.pt"))

# Set model to evaluation mode
model.eval()

# Extract node embeddings
embeddings = extract_node_embeddings(model, graph_pyg)


#----------------------------------
# 3. Test MAB Model with online learning
#----------------------------------


#-----------------------------------------


# Merge embeddings with account labels
embeddings = pd.merge(embeddings, all_accounts, on="account_id", how="left")


# Seperate Classes
df_positive = embeddings[embeddings["is_suspicious"] == 1]
df_negative = embeddings[embeddings["is_suspicious"] == 0]

# Upsample positive class to match negative class size
df_positive_upsampled = resample(df_positive, 
                                 replace=True,     
                                 n_samples=5000, #len(df_negative),   
                                 random_state=187) 

# Combine the upsampled positive class with the negative class
embeddings_2 = pd.concat([df_positive_upsampled, df_negative], ignore_index=True)

# Shuffle to ensure mixed order
embeddings_2 = embeddings_2.sample(frac=1, random_state=187).reset_index(drop=True)

regrets = []
cumulative_regret = []

total_regret = 0

for t, (_, row) in enumerate(embeddings_2.iterrows()):
    context = row.drop(["account_id", "is_suspicious"]).values.astype(float).reshape(1, -1)

    # Predict action using MAB
    action = mab.predict(contexts=context)#[0]

    # True label
    label = row["is_suspicious"]

    # Define rewards
    reward_matrix = {
        ("investigate", 1): 5,
        ("ignore", 1): -3,
        ("investigate", 0): -3,
        ("ignore", 0): 1
    }

    reward_mab = reward_matrix[(action, label)]

    # Determine the best possible reward for this example
    reward_investigate = reward_matrix[("investigate", label)]
    reward_ignore = reward_matrix[("ignore", label)]
    reward_optimal = max(reward_investigate, reward_ignore)

    # Compute regret
    # In your regret calculation
    if label == 1:
        weight = 3  # suspicious cases are more important
    else:
        weight = 1

    regret = weight * (reward_optimal - reward_mab)

    total_regret += regret

    regrets.append(regret)
    cumulative_regret.append(total_regret)

    # Update MAB
    mab.partial_fit(decisions=[action], rewards=[reward_mab], contexts=context)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(cumulative_regret, label="Cumulative Regret")
plt.xlabel("Time step")
plt.ylabel("Cumulative Regret")
plt.title("MAB Cumulative Regret Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.plot(regrets)
plt.title("Instantaneous Regret Over Time")
plt.xlabel("Step")
plt.ylabel("Regret")
plt.grid(True)
plt.show()