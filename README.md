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
- User interface using new simulated data and graphs
- Automatic generation of embeddings and predictions on simulated data
- Dashboard to investigate suspicious accounts and make decisions
- CMAB online learning by updating the pretrained model based on the decisions made during the investigation
- Monitoring tool to assess the performance of the last update as well as the overall model performance

## Project Structure
```
complai/
├── Data/                      
│   ├── Processed/                   # Contains all data and graphs used for model training
│   └── Application_Data/            # Contains the history of investigated accounts and the logo
│ 
├── Model_Development/          
│   ├── data_simulations.py          # Generator for synthetic transaction data
│   ├── train_GNN.py                 # GNN training script
│   ├── train_MAB.py                 # CMAB training script
│   ├── Models/                      # Stores the pretrained model
│   └── utils/
│       ├── utils_data_prep.py       # Utility functions for data generation
│       ├── utils_train_GNN.py       # Utility functions to train the GNN
│       └── utils_train_online_MAB   # Utiltiy functions to train the MAB online
│
├── Application/
│   ├── main.py                      # Script to run the app
│   ├── app.py                       # Streamlit front end for the app
│   └── utils/                 
│       └── utils_app.py             # Utility functions for app the backend
│
└── README.md                  # Project Documentation
``
