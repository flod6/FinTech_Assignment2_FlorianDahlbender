#----------------------------------
# 1. Set Up
#----------------------------------

"""
This script contains all the utility functions and classes used generating
and creating the graph model to generate the embeddings.
"""

# Load Packages
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F

#----------------------------------
# 2. Functions and Objects to train the GNN model
#----------------------------------

# Define Function to convert graph into a PyTorch Geometric Data object
def create_pyg_graph(graph):
    """
    Convertes a NetworkX graph into a PyTorch Geometric Data object.

    Parameters
    graph (networkx.Graph): The input graph.

    Returns
    pyg.data.Data: The PyTorch Geometric Data object.
    """

    # Create a copy of the graph
    g = graph.copy()

    # Convert the data into pyg format
    graph_pyg = pyg.utils.from_networkx(g)

    # Noramlize the node features
    scaler = StandardScaler()
    graph_pyg.x = torch.tensor(scaler.fit_transform(graph_pyg.x), dtype=torch.float32)

    # Return the pyg Graph 
    return graph_pyg


# Create the GNN model
class GraphSage(nn.Module):
    """
    GraphSage model class for node embeddings using PyTorch Geometric. No activation function is applied
    as the model is trained unsupervised and no direct predictions are made.

    Parameters:
    in_channels (int): The number of input features per node.
    hidden_channels (int): The number of hidden units in the hidden layer.
    num_layers (int): The number of layers in the GNN.
    out_channels (int): The number of output features per node.
    dropout (float, optional): Dropout rate for regularization. 
    aggregator (str, optional): The aggregation method to use. 
                                    Defaults to 'mean'.

    Forward Pass:
        Args:
        x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
        edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].

        Returns:
        torch.Tensor: Output node embeddings of shape [num_nodes, out_channels].
    """

    # Set up init method of the class
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int, 
                 num_layers: int,
                 out_channels: int,
                 dropout: float,
                 aggregator: str):
        super(GraphSage, self).__init__()

        # Add dropout Layer
        self.dropout = nn.Dropout(dropout)

        # Create module list for the GraphSage model
        self.convs = nn.ModuleList()

        # Add input layer
        self.convs.append(pyg.nn.SAGEConv(in_channels, hidden_channels, aggr=aggregator))

        # Define Hidden Layer
        for _ in range(num_layers - 1):
            self.convs.append(pyg.nn.SAGEConv(hidden_channels, hidden_channels, aggr=aggregator))

        # Define Output Layer
        self.convs.append(pyg.nn.SAGEConv(hidden_channels, out_channels, aggr=aggregator))

    # Define forward pass of the GNN model
    def forward(self, x, edge_index):
        # Loop over the layer
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = self.dropout(x)
        # Apply the last layer without activation
        x = self.convs[-1](x, edge_index)
        return x


# Define Corruption Function
def corruption(x, edge_index): 
    """
    Introduces randomness to the input data by shuffling the node features.
    To generate more robust representations and is required to the unsupervised
    training of the model.

    Parameters:
    x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
    edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].

    Returns:
    torch.Tensor: Corrupted node feature matrix with the same shape as x.
    """

    # Generate random noise
    noise = torch.randn_like(x) * 0.05

    # Shuffle the node features and add noise
    return x[torch.randperm(x.size(0))] + noise, edge_index


# Define Function to run the epochs
def run_epoch(num_epochs, model, graph, optimizer): 
    """
    Function to run training epochs for the model.

    Parameters:
    num_epochs (int): Number of epochs to train.
    model (torch.nn.Module): The model to train.
    graph (pyg.data.Data): The graph data.
    optimizer (torch.optim.Optimizer): The optimizer for training.

    Returns:
    model (torch.nn.Module): The trained model after the epochs.
    """

    # Iterate over the number of epochs
    for epoch in range(250):

        # Zero the gradients
        optimizer.zero_grad()

        # Create embeddings / Forward pass
        pos_z, neg_z, summary = model(graph.x, graph.edge_index)

        # Derive the loss
        loss = model.loss(pos_z, neg_z, summary)

        # Backpropagation
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Print loss every 10 epochs
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    # Return the model
    return model


# Define a function for the model training process
def train_model(model, graph, num_epochs):

    """ 
    Function to train the GNN model on the provided graph data.
    Note: The model is trained on the full graph using a transductive learning approach as 
    there is only one graph per institution representing the transactions. While for the
    real application dependending on the size of the graph it may be necessary to 
    switch to batch training. 

    Parameters:
    model (torch.nn.Module): The GNN model to be trained.
    graph (torch_geometric.data.Data): The graph data containing node features and edge indices.
    num_epochs (int): The number of epochs to train the model.

    Returns:
    model (torch.nn.Module): The trained GNN model.
    """

    # Set decive 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Send model and graph to device
    model = model.to(device)
    graph = graph.to(device)

    # Define optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Set model to training mode  
    model.train()

    # Train the model 
    run_epoch(250, model, graph, optimizer)

    # Return the trained model
    return model


# Define function to extract the node embeddings
def extract_node_embeddings(model, graph):
    """
    Extracts node embeddings from the trained model.

    Parameters:
    model (torch.nn.Module): The trained GNN model.
    graph (pyg.data.Data): The graph data.

    Returns:
    pd.DataFrame: The node embeddings.
    """

    # Set model to evaluation mode
    model.eval()

    # Extract the node embeddings
    with torch.no_grad():
        node_embeddings = model.encoder(graph.x, graph.edge_index)

    # Extract node IDs 
    node_ids = graph.account_id 
    df = pd.DataFrame(node_embeddings)
    df.insert(0, "account_id", node_ids)

    # Return df with Node embeddings
    return df
