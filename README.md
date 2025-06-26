# FinTech: Business Models and Applications Assignment2 FlorianDahlbender

# ComplAI AML Solutions
ComplAI is a modular FinTech solution that detects accounts involved in money laundering by leveraging **Graph Neural Networks** (GNNs) and **Contextual Multi-Armed Bandits** (CMABs) to identify suspects in real time. Implementing the GNN enables the exploitation of information in transaction networks, while CMAB's online learning capability allows it to adapt to new and evolving money laundering patterns. 

## Features

**Data Simulation**
- Data generator to simulate transactions
- Generator to model the transactions in graph structure
- Two real-world money laundering patterns (layering and smurfing)

**GNN Layer**
- Pretrained GNN model using unsupervised learning
- Embeddings generator for each account

**MAB Layer**
- Pretrained CMAB model trained online
- Classification by leveraging the embeddings

**ComplAI UI Pipeline**
- User interface using new simulated data
- Automatic generation of embeddings and predictions on simulated data
- Dashboard to investigate suspicious accounts and make decisions
- CMAB online learning by updating the model based on the decisions made during the investigation
- Monitoring tool to assess the performance of the last update as well as the overall model performance
