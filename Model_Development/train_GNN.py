#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import pickle
from sklearn.preprocessing import StandardScaler

# Import Custiom Objects and Functions
from Model_Development.utils.utils_train_GNN import (create_pyg_graph, GraphSage, 
                                   corruption, run_epoch, train_model,
                                   extract_node_embeddings)

# Set paths
dir = Path(__file__).resolve().parent.parent

# Set Random Seed
torch.manual_seed(187)

# Load Data
with open(dir / "Data" / "Processed" / "simulated_graph.gpickle", "rb") as f:
    graph = pickle.load(f)
accounts = pd.read_csv(dir / "Data" / "Processed" / "accounts_train.csv")



#----------------------------------
# 1. Create Graph Model
#----------------------------------

# Create the graph in PyTorch Geometric format
graph = create_pyg_graph(graph)

# Create the GrapgSage model
encoder = GraphSage(
    in_channels=graph.num_node_features,
    hidden_channels=64,
    num_layers=5,
    out_channels=32,
    dropout=0,
    aggregator="mean"
)

# Create a Deep Graph Infomax model for unsupervised learning
model = pyg.nn.DeepGraphInfomax(
    hidden_channels=32,
    encoder=encoder,
    summary=lambda z, *args, **kwargs: torch.sigmoid(torch.mean(z, dim=0)),
    corruption=corruption
)

# Set number epochs
num_epochs = 250

# Train the model
model = train_model(model, graph, num_epochs)

# Extract Node Embeddings
embeddings = extract_node_embeddings(model, graph)


# Save the model
torch.save(model.state_dict(), dir / "Model_Development" / "Models" / "gnn_model.pt")

# Save the embeddings
embeddings.to_csv(dir / "Data" / "Processed" / "embeddings_train.csv", index=False)

