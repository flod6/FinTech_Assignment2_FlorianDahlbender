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
- Leverages transactions represented in a graph structure
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

## Requirements
- The MVP was created using Python 3.12
- Required packages are listed in the file `requirements.txt`
- The installation of `torch` and `torch_geometric` must be performed manually, as the correct version depends on the user's system. For details please see official installation instructions:
     - `torch` (https://pytorch.org/get-started/locally/)
     - `torch_geometric` (https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)

## Tech Stack
- **Programming Language**:
     - `Python 3.12`
- **Front End**:
     - `Streamlit` (UI)
- **Back End & ML**:
     - `torch` (GNN)
     - `torch_geometric` (GNN)
     - `mabwiser` (CMAB)
     - `scikit-learn` (Model Evaluation)
- **Utilities**:
     - `pandas` (Data Processing)
     - `numpy` (Data Processing)
     - `matplotlib` (Data Visualization)
     - `networkx` (Graph Creation)

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
│       └── utils_train_online_MAB   # Utility functions to train the MAB online
│
├── Application/
│   ├── main.py                      # Script to run the app
│   ├── app.py                       # Streamlit front end for the app
│   └── utils/                 
│       └── utils_app.py             # Utility functions for the app backend
│
├── requirements.txt                 # Libraries and packages used for the MVP
└── README.md                        # Project Documentation
```

## Installation Guide

1. Clone the Repository
```
git clone https://github.com/flod6/FinTech_Assignment2_FlorianDahlbender.git
cd FinTech_Assignment2_FlorianDahlbender
```

2. Create new Environment
```
conda create -n complai_env python=3.12
conda activate complai_env
```

3. Install Requirements
```bash
pip install -r requirements.txt
```

4. Install `torch` and `torch_geometric` manually, as the version depends on the user's system
- `torch`: (https://pytorch.org/get-started/locally/)
- `torch_geometric`: (https://pytorch-geometric.readthedocs.io/en/2.6.1/install/installation.html)

5. Run the Application
```bash
cd Application
streamlit run app.py
```
Or run the file `Application/main.py` directly in the python editor. 

## Academic Context
This MVP was built as part of an academic assignment for the course FinTech: Business Models & Applications (BM26BAM), for the academic program MSc Business Analytics & Management, at Rotterdam School of Management, Erasmus University. 

## Future Outlook
- Train the models on real transaction data
- Improve GNN architecture and fine-tuning under a supervised setting
- Bank software integration and add pipeline to other departments

## Keywords
**`FinTech`**, **`AML`**, **`Graph Neural Networks`**, **`Multi Armed Bandit`**

## Disclaimer
This is an educational project with simulated data and hypothetical predictions. 

## License 
© 2025 Florian Dahlbender. All rights reserved.

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this software for personal or commercial purposes, provided you include the original copyright and license. 

**Maintainer**: Florian Dahlbender | MSc BAM | RSM


   


