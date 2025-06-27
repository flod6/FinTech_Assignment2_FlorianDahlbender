#----------------------------------
# 1. Set Up
#----------------------------------

"""
This module contains utility functions for the ComplAI application.
It includes functions for loading data, processing inputs, and generating visualizations.
"""

# Load Packages
import pandas as pd
from pathlib import Path
import networkx as nx
import random
import pickle
import torch 
import torch_geometric as pyg
import sys

# Set system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import custom functions
from Model_Development.utils.utils_data_prep import generate_data, create_graph, extract_all_accounts
from Model_Development.utils.utils_train_GNN import create_pyg_graph
from Model_Development.utils.utils_train_online_MAB import reward_function
from Model_Development.utils.utils_train_GNN import GraphSage, corruption

# Set paths
dir = Path(__file__).resolve().parent.parent.parent

# Set random seed for reproducibility
random.seed(6)
torch.manual_seed(6)

#----------------------------------
# 2. Define Function to Run the App
#---------------------------------- 


# Define function to simulate new data for the application
def generate_test_data(): 
    """
    Generates a DataFrame with random data for testing purposes.

    Parameters:
    None
    
    Returns:
        pd.DataFrame: A DataFrame with random values.
    """

    # Set random seed for reproducibility
    random.seed(6)

    # Set Parameter for data generation
    n_accounts = 2500
    n_transactions = 12500

    # Define Number of Layering Chains and Length of Each Chain
    n_chains = 3
    chain_length = 5

    # Define Number of Accounts to Simulate Smurfing
    n_smurfing = 3
    num_smurfing_deposits = 5

    # Create DataFrame with Simulated Transactions
    df = generate_data(n_accounts, n_transactions, n_chains, chain_length, n_smurfing, num_smurfing_deposits, seed=6)

    # Extract all accounts from the DataFrame
    accounts = extract_all_accounts(df)

    # Create a graph from the DataFrame
    graph = create_graph(df)

    # Return Items
    return df, graph, accounts


# Define function to create embedding for the graph
def generate_test_embeddings(graph):
    """
    Generates a random embedding for the provided graph.
    
    Parameters:
    graph (networkx.Graph): The graph for which to generate embeddings.
    
    Returns:
    embeddings (pd.Dataframe): A dataframe that contains the embeddings for each node in the graph
    """

    # Create PyTorch Geometric graph
    graph_pyg = create_pyg_graph(graph)

    # Set up the GNN model that was employed in the Model Development phase
    encoder = GraphSage(
        in_channels=graph_pyg.num_node_features,
        hidden_channels=64,
        num_layers=5,
        out_channels=32,
        dropout=0,
        aggregator="mean"
    )
    model = pyg.nn.DeepGraphInfomax(
        hidden_channels=32,
        encoder=encoder,
        summary=lambda z, *args, **kwargs: torch.sigmoid(torch.mean(z, dim=0)),
        corruption=corruption
    )
    
    # Load the GNN
    model.load_state_dict(torch.load(dir / "Model_Development" / "Models" / "gnn_model.pt"))
    
    # Set model to evaluation mode
    model.eval()
    
    # Extract the node embeddings
    with torch.no_grad():
        node_embeddings = model.encoder(graph_pyg.x, graph_pyg.edge_index)

    # Extract node IDs and append to embeddings
    node_ids = graph_pyg.account_id 
    embeddings = pd.DataFrame(node_embeddings)
    embeddings.insert(0, "account_id", node_ids)

    # Return the embeddings
    return embeddings


# Define Function to make predictions
def make_test_predictions(embeddings):
    """
    Makes predictions based on the provided embeddings.
    
    Parameters:
    embeddings (pd.DataFrame): DataFrame containing node embeddings.
    
    Returns:
    predictions pd.DataFrame: DataFrame with predictions added.
    """
    
    # Load the mab model from the development phase
    with open(dir / "Model_Development" / "Models" / "mab_model.pkl", "rb") as f:
        mab = pickle.load(f)

    # Make predictions using the mab model
    predictions = mab.predict(embeddings.drop(columns=["account_id"]).values)

    # Return the predictions
    return predictions


# Define functions to visualize for investigate page
def extract_suspicious_accounts(df, graph, accounts, embeddings, predictions): 

    """
    Extracts suspicious accounts from the graph and returns a DataFrame with predictions.

    Parameters:
    df (pd.DataFrame): DataFrame containing transaction data.
    graph (networkx.Graph): Graph representation of the transactions.
    accounts (list): List of account IDs.
    embeddings (pd.DataFrame): DataFrame containing node embeddings.
    predictions (list): List of predictions for each account.

    Returns:
    flagged_df (pd.DataFrame): DataFrame containing flagged accounts and their predictions.
    """


    # Call created functions
    embeddings = generate_test_embeddings(graph)
    predictions = make_test_predictions(embeddings)

    # Merge embeddings with predictions
    embeddings["predictions"] = predictions

    # Extract all flagged accounts
    flagged_accounts = embeddings[embeddings["predictions"] == "investigate"]

    # Create DF that is displayed
    flagged_df = pd.DataFrame(flagged_accounts, columns=["account_id", "predictions"])

    # Rename columns
    flagged_df.rename(columns={"account_id": "Account ID", "predictions": "Prediction"}, inplace=True)

    # Return the flagged DataFrame
    return flagged_df


# Define function to make elements for investige page
def create_investigate_elements(flagged_df, selected_account, transactions):
    """
    Creates Streamlit elements for the investigate page.

    Parameters:
    flagged_df (pd.DataFrame): DataFrame containing flagged accounts.
    selected_account (str): Account ID of the selected account for investigation.
    transactions (pd.DataFrame): DataFrame containing transaction data.
    
    Returns:
    node_features (pd.DataFrame): DataFrame containing node-level features.
    sent_transactions (pd.DataFrame): DataFrame containing transactions sent by the selected account.
    received_transactions (pd.DataFrame): DataFrame containing transactions received by the selected account.
    G (networkx.Graph): Graph representation of the transactions for the selected account.
    """

    # Filter transactions for the selected account
    account_transactions = transactions[
        (transactions["sender"] == selected_account) |
        (transactions["receiver"] == selected_account)
    ]

    # Calculate node-level features
    total_sent = account_transactions[account_transactions["sender"] == selected_account]["amount"].sum()
    total_received = account_transactions[account_transactions["receiver"] == selected_account]["amount"].sum()
    num_sent = (account_transactions["sender"] == selected_account).sum()
    num_received = (account_transactions["receiver"] == selected_account).sum()
    avg_amount_sent = account_transactions[account_transactions["sender"] == selected_account]["amount"].mean()
    avg_amount_received = account_transactions[account_transactions["receiver"] == selected_account]["amount"].mean()
    std_amount_sent = account_transactions[account_transactions["sender"] == selected_account]["amount"].std()
    std_amount_received = account_transactions[account_transactions["receiver"] == selected_account]["amount"].std()
    unique_partners_sent = account_transactions[account_transactions["sender"] == selected_account]["receiver"].nunique()
    unique_partners_received = account_transactions[account_transactions["receiver"] == selected_account]["sender"].nunique()

    # Create a feature summary
    node_features = pd.DataFrame({
        "Metric": [
            "Total Sent", "Total Received",
            "Number of Transactions Sent", "Number of Transactions Received", 
            "Average Amount Sent", "Average Amount Received",
            "Standard Deviation Amount Sent", "Standard Deviation Amount Received",
            "Unique Partners Sent", "Unique Partners Received"
        ],
        "Value": [
            total_sent, total_received,
            num_sent, num_received,
            avg_amount_sent, avg_amount_received,
            std_amount_sent, std_amount_received,
            unique_partners_sent, unique_partners_received
        ]
    })

    # Filter all transactions sent by the selected account
    sent_transactions = account_transactions[account_transactions["sender"] == selected_account]
    sent_transactions = sent_transactions.drop(columns=["is_suspicious"])

    # Rename columns for better readability
    sent_transactions = sent_transactions.rename(columns={
        "sender": "Sender",
        "receiver": "Receiver",
        "amount": "Amount",
        "timestamp": "Timestamp"
    })

    # Filter all transactions received by the selected account
    received_transactions = account_transactions[account_transactions["receiver"] == selected_account]
    received_transactions = received_transactions.drop(columns=["is_suspicious"])

    # Rename columns for better readability
    received_transactions = received_transactions.rename(columns={
        "sender": "Sender",
        "receiver": "Receiver",
        "amount": "Amount",
        "timestamp": "Timestamp"
    })

    # Extract all transactions to create a network graph for the selected account
    G = nx.from_pandas_edgelist(
        account_transactions,
        source="sender",
        target="receiver",
        edge_attr="amount",
        create_using=nx.DiGraph())

    # Return the elements
    return node_features, sent_transactions, received_transactions, G


# Create function to update the mab model
def update_model(embeddings, investigated_accounts):

    """
    Updates the Multi-Armed Bandit (MAB) model with new embeddings and suspicious accounts.

    Parameters:
    embeddings (pd.DataFrame): DataFrame containing node embeddings.
    investigated_accounts (pd.DataFrame): DataFrame containing accounts that have been investigated.
    
    Returns:
    df (pd.DataFrame): DataFrame with updated rewards and actions for all flagged accounts.
    """

    # Load the mab model
    with open(dir / "Model_Development" / "Models" / "mab_model.pkl", "rb") as f:
        mab = pickle.load(f)

    # Filter out investigated accounts from the embeddings
    df = embeddings[embeddings["account_id"].isin(investigated_accounts["Account"])]

    # Join the dataframes
    df = df.merge(investigated_accounts, left_on="account_id", right_on="Account", how="left")

    # Rename columns
    df.rename(columns={"Decision": "is_suspicious", "Prediction": "action_taken"}, inplace=True)
    
    # Change values in column Decision to match the reward function
    df["is_suspicious"] = df["is_suspicious"].replace({"Fraud": 1, "Not Fraud": 0})

    # Retrive Reward
    df["reward"] = df.apply(reward_function, axis=1)

    # Retrieve context, reward, and action
    contexts = df.drop(columns=["account_id", "is_suspicious", "action_taken", "reward", "Account"]).values
    rewards = df["reward"].values
    actions = df["action_taken"].values

    # Update the MAB model with the new data
    mab.partial_fit(decisions=actions, rewards=rewards, contexts=contexts)

    # Save the updated model
    with open(dir / "Model_Development" / "Models" / "mab_model.pkl", "wb") as f:
        pickle.dump(mab, f)

    # Return df with all the results
    return df[["account_id", "is_suspicious", "action_taken", "reward"]]



# Define function to get the monitoring results
def monitor_items(results): 
    """
    Create items that are used during the monitoring phase of the app.

    Parameters:
    results (pd.DataFrame): DataFrame containing the results of the monitoring phase.

    Returns:
    correct_predictions (pd.DataFrame): DataFrame containing correctly predicted accounts.
    incorrect_predictions (pd.DataFrame): DataFrame containing incorrectly predicted accounts.
    overall_results (pd.DataFrame): DataFrame containing the overall results of the monitoring phase.
    """

    # Map action_taken to numeric predictions
    action_map = {"investigate": 1, "ignore": 0}
    results["predicted_label"] = results["action_taken"].map(action_map)

    # Filter all correctly predicted accounts
    correct_predictions = results[results["is_suspicious"] == results["predicted_label"]]

    # Make data frame more fancy
    correct_predictions = correct_predictions.rename(columns={
        "account_id": "Account ID",
        "is_suspicious": "Actual Label",
        "predicted_label": "Predicted Label",
        "action_taken": "Action Taken",
        "reward": "Reward"
    })

    # Filter all incorrectly predicted accounts
    incorrect_predictions = results[results["is_suspicious"] != results["predicted_label"]]

    # Make data frame more fancy
    incorrect_predictions = incorrect_predictions.rename(columns={
        "account_id": "Account ID",
        "is_suspicious": "Actual Label",
        "predicted_label": "Predicted Label",
        "action_taken": "Action Taken",
        "reward": "Reward"
    })

    # Check if the overall summary already file exists if now a new one is created
    file_path = dir / "Data" / "Application_Data" / "monitoring_results.csv"
    if file_path.exists():
        overall_results = pd.read_csv(file_path)
    else:
        overall_results = pd.DataFrame(columns=["Account ID", "Actual Label", "Predicted Label", "Action Taken", "Reward", "Run"])

    # Add a new column for the run number
    if not overall_results.empty:
        run_number = overall_results["Run"].max() + 1
    else:
        run_number = 1

    # Append the number of the run
    correct_predictions["Run"] = run_number
    incorrect_predictions["Run"] = run_number

    # Append the new results to the overall results
    overall_results = pd.concat([overall_results, correct_predictions], ignore_index=True)
    overall_results = pd.concat([overall_results, incorrect_predictions], ignore_index=True)

    # Store the overall results
    overall_results.to_csv(file_path, index=False)
    
    # Return the items
    return correct_predictions, incorrect_predictions, overall_results


# Make function to get the overall results if now accounts has been investigated yet
def get_overall_results():

    # Load the file with the overall results
    file_path = dir / "Data" / "Application_Data" / "monitoring_results.csv"
    if file_path.exists():
        overall_results = pd.read_csv(file_path)
    else:
        overall_results = pd.DataFrame(columns=["Account ID", "Actual Label", "Predicted Label", "Action Taken", "Reward", "Run"])

    # Return the data frame
    return overall_results


# Define function to get the items required for the monitoring plot
def monitor_plot(overall_results):

    """
    Define function to get metrics for the monitoring plot

    Parameters:
    overall_results (pd.DataFrame): DataFrame containing the overall results of the monitoring phase.

    Returns:
    runs (np.ndarray): Unique run numbers.
    accuracy_per_run (list): List of accuracy values for each run.
    false_positive_rate_per_run (list): List of false positive rates for each run.
    """

    # Extract performance metrics over time
    runs = overall_results["Run"].unique()
    accuracy_per_run = []
    false_positive_rate_per_run = []

    # Calculate accuracy and false positive rate for each run
    for run in runs:
        run_data = overall_results[overall_results["Run"] == run]
        accuracy = (run_data["Actual Label"] == 1).sum() / len(run_data)
        false_positive_rate = (run_data["Actual Label"] == 0).sum() / len(run_data)
        accuracy_per_run.append(accuracy)
        false_positive_rate_per_run.append(false_positive_rate)

    # Return Items
    return runs, accuracy_per_run, false_positive_rate_per_run



