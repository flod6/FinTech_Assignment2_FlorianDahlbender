#----------------------------------
# 1. Set Up
#----------------------------------
"""
This script contains all the utility functions used for the 
initial data prepreration steps in the project.
"""

# Load Packages
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


#----------------------------------
# 2. Data Simulation Functions and Objects
#----------------------------------


# Define Function to Generate Simulated Transaction Data
def generate_data(n_accounts, n_transactions, n_chains, chain_length, n_smurfing, num_smurfing_deposits, seed=187):

    """
    Generates a DataFrame simulating transactions between accounts with 
    layering and smurfing activities.
    
    Parameters:
    n_accounts (int): Number of unique accounts.
    n_transactions (int): Number of transactions to simulate.
    n_chains (int): Number of layering chains to create.
    chain_length (int): Length of each layering chain.
    n_smurfing (int): Number of accounts to simulate smurfing.
    num_smurfing_deposits (int): Number of smurfing deposits per account.
    seed (int): Random seed for reproducibility.
    
    Returns:
    pd.DataFrame: A DataFrame containing simulated transaction data including layering and smurfing transactions.
    """

    # Set seed for reproducibility
    np.random.seed(seed)
    
    # Generate random account IDs
    accounts = [f"A_{i}" for i in range(n_accounts)]
    
    # Randomly select senders, receivers, amounts, and timestamps
    senders = np.random.choice(accounts, n_transactions)
    receivers = np.random.choice(accounts, n_transactions)
    amounts = np.random.exponential(scale=1000, size=n_transactions) 
    timestamps = (pd.date_range("2025-06-01", periods=n_transactions, freq='min') + pd.to_timedelta(
                 np.random.randint(0, 3600, n_transactions), unit='s'))
    
    # Create DataFrame by merging items together
    df = pd.DataFrame({
        "sender": senders,
        "receiver": receivers,
        "amount": amounts,
        "timestamp": timestamps,
        "is_suspicious": 0  # Placeholder for suspicious flag
    })


    # Add Layering to the transactions
    df = add_layering(df, accounts, timestamps, n_chains, chain_length)

    # Add Smurfing to the transactions
    df = add_smurfing(df, accounts, timestamps, n_smurfing, num_smurfing_deposits)

    # Drop Accounts that send money to themselves
    df = df[df["sender"] != df["receiver"]] 
    
    # Return transaction data DataFrame
    return df



# Define Function to Add Layering to Transactions
def add_layering(df, accounts, timestamps, n_chains, chain_length):

    """
    Adds layering to the transaction data by creating chains of transactions

    Parameters:
    df (pd.DataFrame): The original transaction DataFrame.
    accounts (list): List of account IDs.
    timestamps (pd.DatetimeIndex): Timestamps for the transactions.
    n_chains (int): Number of layering chains to create.
    chain_length (int): Length of each layering chain.

    Returns:
    pd.DataFrame: A DataFrame with added layering transactions.
    
    """

    # Iterate over the number of chains in the data
    for i in range(n_chains):

        # Randomly select accounts for the chain
        chain = np.random.choice(accounts, chain_length, replace=False)
        base_time = np.random.choice(timestamps)

        # Define time offset
        time_offset = 0

        # Iterate over the chain length 
        for j in range(chain_length - 1):

            # Define time offset for the transaction
            time_offset += np.random.randint(1, 4)

            df = pd.concat([df, pd.DataFrame({
                "sender": [chain[j]],
                "receiver": [chain[j + 1]],
                "amount": [10000 + np.random.normal(0, 200)], # 500
                "timestamp": [base_time + pd.Timedelta(hours=time_offset)],
                "is_suspicious": [1]  # Flag as suspicious
                })])
            

        # Add non-suspicious transactions to these accounts
        for k in chain:

            # Define number of noise transactions
            n_noise = np.random.randint(2, 5)

            # Generate noise transactions
            for _ in range(n_noise):

                # Randomly select account for noise transaction
                partner = np.random.choice(accounts)

                # Checke that the partner is not the same as the current account
                if partner == k:
                    continue

                # Define the direction of the transaction
                direction = np.random.choice(["send", "receive"])
                sender, receiver = (k, partner) if direction == "send" else (partner, k)
                noise_time = base_time + pd.Timedelta(hours=np.random.randint(1, 10))

                # Append the noise transaction to the DataFrame
                df = pd.concat([df, pd.DataFrame({
                    "sender": [sender],
                    "receiver": [receiver],
                    "amount": [np.random.exponential(scale=100)],
                    "timestamp": [noise_time],
                    "is_suspicious": [0]  # Not flagged as suspicious
                    })])
            
    # Return the DataFrame with layering transactions added
    return df



# Define Function to add smurfing
def add_smurfing(df, accounts, timestamps, n_smurfing, num_smurfing_deposits):

    """
    Adds smurfing transactions to the DataFrame by creating small transactions between accounts.

    Parameters:
    df (pd.DataFrame): The original transaction DataFrame.
    accounts (list): List of account IDs.
    timestamps (pd.DatetimeIndex): Timestamps for the transactions.
    n_smurfing (int): Number of accounts to simulate smurfing.
    num_smurfing_deposits (int): Number of smurfing deposits per account.

    Returns:
    pd.DataFrame: A DataFrame with added smurfing transactions.
    """

    # Define target accounts that receive smurfing 
    targets = np.random.choice(accounts, n_smurfing, replace=False)

    # Iterate over the target accounts
    for target in targets:

        # Set base time
        base_time = np.random.choice(timestamps)

        # Iterate over the number of smurfing deposits
        for _ in range(num_smurfing_deposits):

           # for _ in range(np.random.randint(10, 15)):

            # Randomly select sender accounts
            sender = np.random.choice(accounts)

            # Set deposit time
            deposit_time = base_time + pd.Timedelta(minutes=np.random.randint(0, 3))

            # Append the smurfing transaction to the DataFrame
            df = pd.concat([df, pd.DataFrame({
                    "sender": [sender],
                    "receiver": [target],
                    "amount": [950 + np.random.normal(0, 100)], 
                    "timestamp": [deposit_time],
                    "is_suspicious": [1] # Flag as suspicious
            })])

        # Add noise to the smurfing transactions by letting the accounts receive money from other accounts
        # and let the target send money to other accounts

        # Generate number of noise transactions
        n_noise = np.random.randint(5, 10)

        # Iterate over the noise transactions
        for _ in range(n_noise):

            # Randomly select a partner account
            partner = np.random.choice(accounts)

            # Check that the partner is not the same as the target account
            if partner == target:
                continue

            # Define the direction of the transaction
            direction = np.random.choice(["send", "receive"])
            sender, receiver = (target, partner) if direction == "send" else (partner, target)

            # Add time
            noise_time = base_time + pd.Timedelta(minutes=np.random.randint(10, 90))

            # Append the noise transaction to the DataFrame
            df = pd.concat([df, pd.DataFrame({
                "sender": [sender],
                "receiver": [receiver],
                "amount": [np.random.exponential(scale=500)],
                "timestamp": [noise_time + pd.Timedelta(hours=np.random.randint(1, 150))],
                "is_suspicious": [0]  # Not flagged as suspicious
            })])
            
    # Return the DataFrame with smurfing transactions added
    return df


# Define a function to extract all accounts and their labels
def extract_all_accounts(df):
    """
    Extract all acounts and their respective labels. 

    Parameters:
    df (pd.DataFrame): DataFrame containing transaction data with 'is_suspicious' column.

    Returns:
    pd.DataFrame: DataFrame with unique accounts and their labels indicating if they are suspicious.    
    """
    # Get all suspicious accounts
    suspicious_accounts = set(df[df["is_suspicious"] == 1]["receiver"])

    # Extract all unique accounts from the DataFrame
    all_accounts = set(df["sender"]) | set(df["receiver"])

    # Create a DataFrame with all accounts and their labels
    accounts_df = pd.DataFrame({
        "account_id": list(all_accounts),
        "is_suspicious": [1 if account in suspicious_accounts else 0 for account in all_accounts]
    })

    # Return all the suspicious accounts
    return accounts_df


#----------------------------------
# 3. Graph Creation Functions and Objects
#----------------------------------


# Define function to create a graph from the transaction data frame
def create_graph(df):
    """
    Creates a directed graph from the transaction DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing transaction data.
    
    Returns:
    nx.DiGraph: A directed graph representing the transactions.
    """


    # Min Max normalize the timestamps
    df["timestamp"] = (df["timestamp"] - df["timestamp"].min()) / (df["timestamp"].max() - df["timestamp"].min())

    # Create empty directed graph object
    G = nx.DiGraph()

    # Extract Unique accounts to represent nodes
    accounts = pd.unique(df[["sender", "receiver"]].values.ravel())

    # Add the accounts as nodes to the graph
    G.add_nodes_from(accounts)

    # Add account ID as to the nodes
    for i, account in enumerate(G.nodes()):
        G.nodes[account]["account_id"] = account

    # Add edges to the graph for each transaction

    # Iterate over the rows to add edges
    for _, row in df.iterrows():
        
        # Add edge from sender to receiver with the transactions information as
        # edge features
        edge_features = [float(row["amount"]), float(row["timestamp"])]

        G.add_edge(
            row["sender"], 
            row["receiver"],
            edge_attr = edge_features
        )

    
    ## Add node attributes for all the accounts

    # Get node level features
    features_df = pd.DataFrame(index=accounts)
    features_df["total_sent"] = df.groupby("sender")["amount"].sum()
    features_df["total_received"] = df.groupby("receiver")["amount"].sum()
    features_df["transactions_sent"] = df.groupby("sender").size()
    features_df["transactions_received"] = df.groupby("receiver").size()
    features_df["avg_amount_sent"] = df.groupby("sender")["amount"].mean()
    features_df["avg_amount_received"] = df.groupby("receiver")["amount"].mean()
    features_df["std_amount_sent"] = df.groupby("sender")["amount"].std()
    features_df["std_amount_received"] = df.groupby("receiver")["amount"].std()
    features_df["unique_partners_sent"] = df.groupby("sender")["receiver"].nunique()
    features_df["unique_partners_received"] = df.groupby("receiver")["sender"].nunique()
    features_df = features_df.fillna(0)


    # Append features to the graph nodes
    for account in G.nodes():
        G.nodes[account]["x"] = features_df.loc[account].values.astype(np.float32)

    # Return the graph object
    return G



# Define Function for some quick sanity checks
def graph_sanity_check(graph, df):

    """
    Performs basic sanity checks on the transaction graph to ensure its integrity
    by printing some basic statistics of the graph.

    Parameters:
    graph (networkx.Graph): The transaction graph to be checked.
    df (pd.DataFrame): The DataFrame containing transaction data.

    Returns:
    None
    """

    # Print basic statistics of the graph
    print("Basic Graph Statistics:")
    print(f"Number of Nodes: {graph.number_of_nodes()}")
    print(f"Number of Edges: {graph.number_of_edges()}")
    print(f"Total Transactions: {len(df)}")
    print("-" * 40)

    # Check for node presence in the graph
    missing_nodes = 0
    all_accounts = set(df["sender"]) | set(df["receiver"]) 

    # Iterate over all accounts to check presence 
    for account in all_accounts:
        if account not in graph:
            missing_nodes += 1
    print(f"Missing Nodes: {missing_nodes} out of {len(all_accounts)}")

    # Print missing nodes if present
    if missing_nodes > 0:
        print("Missing Nodes:", [node for node in all_accounts if node not in graph])

    print("-" * 40)

    # Sample some node statistics
    sample_node = next(iter(graph.nodes()))
    print(f"Sample Node: {sample_node}")
    print(f"Attributes: {graph.nodes[sample_node]}")
    print("-" * 40)

    # Sample some edge statistics
    sample_edge = next(iter(graph.edges()))
    print(f"Sample Edge: {sample_edge}")
    print(f"Attributes: {graph.edges[sample_edge]}")




# Define Function to Plot Graph
def plot_graph(graph, df, limit_nodes):

    """
    Plot a directed graph with nodes representing accounts and edges representing transactions.

    Parameters:
    graph (networkx.DiGraph): The directed graph to plot.
    df (pd.DataFrame): DataFrame containing transaction data for labeling nodes.

    Returns:
    None: Displays the graph plot.
    """

    # Identify suspicoius accounts
    suspicious_accounts = set(df[df["is_suspicious"] == 1]["sender"]) | \
                          set(df[df["is_suspicious"] == 1]["receiver"])
    
    print(len(suspicious_accounts), "suspicious accounts identified.")


    # Create Subgraph if necessary
    if limit_nodes is not None and graph.number_of_nodes() > limit_nodes:
        sampled_nodes = list(list(graph.nodes)[:limit_nodes])
        graph = graph.subgraph(sampled_nodes).copy()

    # Define Layout
    pos = nx.spring_layout(graph, seed=187)

    # Define node colors to highlight suspicious accounts
    node_colors = ['red' if node in suspicious_accounts else "lightblue" for node in graph.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color=node_colors, alpha=0.8)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, arrowstyle='-|>', arrowsize=10, edge_color='gray', alpha=0.5)

    # Draw labels
    # nx.draw_networks_labels(graph, pos, font_size=8)

    # Define reamining parameters
    plt.title("Transaction Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

