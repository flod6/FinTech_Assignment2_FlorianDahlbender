#----------------------------------
# 1. Set Up
#----------------------------------

"""
This script contains all the utility functions and classes used generating
and creating and training the Multi-Armed Bandit (MAB) model.
"""

# Load Packages 
import pandas as pd
import numpy as np
from mabwiser.mab import MAB, LearningPolicy
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt

#----------------------------------
# 2. Define Functions and Objects
#----------------------------------

# Define Reward function 
def reward_function(row):
    """
    Reward function to evaluate the actions taken on accounts and provide the MAB
    with feedback based on the action taken and whether the account is suspicious.
    The reward are constructed in a way that reflect the economic costs of the decision and
    to tune the MAB in a meaniful way. For instance, missing a suspicious account, is more 
    costly than investigating a non-suspicious account due to high fines.

    Parameters:
    row (pd.Series): A row from the DataFrame containing account information.

    Returns:
    int: A reward value based on the action taken and whether the account is suspicious.
    """

    # Positive reward for investigating suspicious accounts
    if row["is_suspicious"] == 1 and row["action_taken"] == "investigate":
        return 5 
    # No Reward for ignoring non-suspicious accounts
    elif row["is_suspicious"] == 0 and row["action_taken"] == "ignore":
        return 0
    # Negative reward for ignoring suspicious accounts
    elif row["is_suspicious"] == 1 and row["action_taken"] == "ignore":
        return -5
    # Negative reward for investigating non-suspicious accounts
    elif row["is_suspicious"] == 0 and row["action_taken"] == "investigate":
        return -0.5


# Define Function to set up MAB model
def setup_mab_model(arms, policy, explore, seed, cont):

    """
    Set up the MAB model with the specified arms and policy.

    Parameters:
    arms (list): List of arms to be used in the MAB model to reflect the decisions.
    policy (str): The learning policy to be used. LinUCB or LinTS possible
    explore (float): The exploration rate for the MAB model.
    seed (int): Random seed for reproducibility.
    cont (np.ndarray): Context features for the MAB model.

    Returns:
    MAB: Configured MAB model.
    """
    
    # Define policy map
    policy_map = {
        "linucb": LearningPolicy.LinUCB(alpha=explore),
        "lints": LearningPolicy.LinTS(alpha=explore)
    }

    # Define Model
    mab = MAB(arms=arms, learning_policy=policy_map[policy], seed=seed)

    # Fit the MAB model with empty decisions and rewards
    empty_contexts = np.empty((0, cont.shape[1]))
    mab.fit(decisions=[], rewards=[], contexts=empty_contexts)

    # Return the MAB model
    return mab


# Define Funtion for the MAB training Loop
def training_loop(mab, X, y, arms):
    """
    Training loop for the Multi-Armed Bandit (MAB) model.

    Parameters:
    mab (MAB): The Multi-Armed Bandit model to be trained.
    X (np.ndarray): The context data for the MAB model.
    y (np.ndarray): The labels corresponding to the context data.
    arms (list): List of arms available in the MAB model.

    Returns:
    mab (MAB): The trained Multi-Armed Bandit model.
    history_df (pd.DataFrame): DataFrame containing the training history.
    """

    # Define items to track the results
    history = []
    cumulative_rewards = 0
    cumulative_regret = 0

    # Loop through each account
    for i in range(len(X)):

        # Get the context and label for the current account
        context = X[i]
        label = y[i]

        # Choose an arm and action
        choosen_arm = mab.predict([context])

        # Define the reward
        reward = reward_function(pd.Series({
            "is_suspicious": label,
            "action_taken": choosen_arm
        }))

        # Get optimal reward that could be achieved
        optimal_reward = max([reward_function(pd.Series({
            "is_suspicious": label,
            "action_taken": arm
        })) for arm in arms])
        
        # Update the MAB model based on the reward and decision
        mab.partial_fit(decisions=[choosen_arm], rewards=[reward], contexts=[context])

        # Track Metrics
        cumulative_rewards += reward
        cumulative_regret += (optimal_reward - reward)

        # Update history
        history.append({
            "iteration": i,
            "chosen_arm": choosen_arm,
            "true_label": label,
            "reward": reward,
            "optimal_reward": optimal_reward,
            "regret": (optimal_reward - reward),
            "cumulative_reward": cumulative_rewards,
            "cumulative_regret": cumulative_regret
        })

    # Save training history
    history_df = pd.DataFrame(history)

    # Return model and history of decisions
    return mab, history_df

# Define Function to generate plots for MAB training history
def generate_plots(history_df): 

    """
    Generate plots to visualize the performance of the MAB model over time.

    Parameters:
    history_df (pd.DataFrame): DataFrame containing the history of rewards and actions taken.

    Returns:
    None: Displays plots of cumulative reward, cumulative regret, average regret, and average reward.
    """

    # Plot Cumulative Regret
    plt.figure(figsize=(6, 6))
    plt.plot(history_df["iteration"], history_df["cumulative_regret"], label="Cumulative Regret", color="orange")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Cumulative Reward and Regret Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Cumulative Reward
    plt.figure(figsize=(6, 6))
    plt.plot(history_df["iteration"], history_df["cumulative_reward"], label="Cumulative Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Cumulative Reward and Regret Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate average regret per period
    history_df["average_regret"] = history_df["cumulative_regret"] / (history_df["iteration"] + 1)

    # Plot average regret over time
    plt.figure(figsize=(10, 8))
    plt.plot(history_df["iteration"], history_df["average_regret"], label="Average Regret", color="lightgreen")
    plt.xlabel("Iteration")
    plt.ylabel("Average Regret")
    plt.title("Average Regret Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate average reward per period
    history_df["average_reward"] = history_df["cumulative_reward"] / (history_df["iteration"] + 1)

    # Plot average regret over time
    plt.figure(figsize=(10, 8))
    plt.plot(history_df["iteration"], history_df["average_reward"], label="Average Reward", color="blue")
    plt.xlabel("Iteration")
    plt.ylabel("Average Reward")
    plt.title("Average Reward Over Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Define Funtion to evaluate the last 1000 observations
def evaluate_MAB(history_df, last_n=1000):

    """
    Evaluate the MAB model on the last 1000 observations.

    Parameters:
    history_df (pd.DataFrame): DataFrame containing the training history.
    mab (MAB): The trained Multi-Armed Bandit model.

    Returns:
    None: Prints the evaluation metrics and confusion matrix.
    """ 

    # Get the last 1000 observations
    recent_history = history_df.tail(last_n)

    # Extract true labels and predicted actions
    true_labels = recent_history["true_label"].values
    predicted_actions = recent_history["chosen_arm"].apply(lambda x: 1 if x == "investigate" else 0).values

    # Calculate accuracy, confusion matrix, and classification report
    accuracy = accuracy_score(true_labels, predicted_actions)
    conf_matrix = confusion_matrix(true_labels, predicted_actions)
    report = classification_report(true_labels, predicted_actions, digits=3, output_dict=True)

    # Print the evaluation results
    print(f"Accuracy: {accuracy:.3f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(conf_matrix,
                       index=["Actual Non-Suspicious", "Actual Suspicious"],
                       columns=["Predicted Non-Suspicious", "Predicted Suspicious"]))
    print("\nClassification Report:")
    print(pd.DataFrame(report).T[["precision", "recall", "f1-score", "support"]])
