#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Import Custom Objects and Functions
from Model_Development.utils.utils_train_online_MAB import (setup_mab_model, 
                                                            training_loop, generate_plots, 
                                                            evaluate_MAB)

# Set paths
dir = Path(__file__).resolve().parent.parent

# Load Data
embeddings = pd.read_csv(dir / "Data" / "Processed" / "embeddings_train.csv")
accounts = pd.read_csv(dir / "Data" / "Processed" / "accounts_train.csv")

# Set Random Seed
np.random.seed(187)

#----------------------------------
# 2. Set Up MAB
#----------------------------------

# Merge Data to add the labels to the embeddings
embeddings = pd.merge(embeddings, accounts, on="account_id", how="left")

# Define the arms
arms = ["investigate", "ignore"]

# Extract context for the MAB
X = embeddings.drop(columns=["account_id", "is_suspicious"]).values

# Extract labels for the MAB
y = embeddings["is_suspicious"].values

# Set up the MAB model
mab = setup_mab_model(arms=arms, policy="lints", explore=0.2, seed=187, cont=X)

# Train the MAB in an online fashion
mab_trained, history_df = training_loop(mab, X, y, arms)


#----------------------------------
# 3. Evaluate the MAB
#----------------------------------

# Generate Plots
generate_plots(history_df)

# Generate Confusion Matrix and Test Statistics
evaluate_MAB(history_df)

# Save the trained model
with open(dir / "Model_Development" / "Models" / "mab_model.pkl", "wb") as f:
    pickle.dump(mab_trained, f)

