
#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import streamlit as st
from PIL import Image
from pathlib import Path
import pickle
import pandas as pd
import sys

# Set system path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import custom functions
from Application.utils.utils_app import *

# Set paths
dir = Path(__file__).resolve().parent.parent

# Set random seed for reproducibility
random.seed(6)


#----------------------------------
# 2. Create Streamlit App
#----------------------------------

# Set page configuration
st.set_page_config(
    page_title="ComplAI – AML Review",
    page_icon="🔍",
    layout="centered"
)

# Load the Image
logo = Image.open(dir / "Data" / "Application_Data" / "ComplAI_logo.png")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "title"

# Convenience variable
page = st.session_state.page


#----------------------------------
# 3. Header
#----------------------------------

col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=100)
with col2:
    st.title("ComplAI – AML Review")

# Optional styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f5;
    }
    </style>
    """,
    unsafe_allow_html=True
)


#----------------------------------
# 4. Title Page
#----------------------------------

if page == "title":
    st.markdown(
        """
        ## Welcome to the AML Review Application
        This application is designed to help you review suspicious accounts detected by the model.
        Click the button below to begin your review.
        """
    )

    if st.button("Start Review"):
        st.session_state.page = "review"
        st.rerun()

    # Generate data for the demo
    df, graph, accounts = generate_test_data()

    # Store data is the session state for later use
    st.session_state.df = df
    st.session_state.graph = graph
    st.session_state.accounts = accounts

    # Generate embeddings for the demo
    embeddings = generate_test_embeddings(graph, accounts)

    # Store embeddings in the session state for later use
    st.session_state.embeddings = embeddings

    # Make predictions using the model
    predictions = make_test_predictions(embeddings)

    # Store predictions in the session state for later use
    st.session_state.predictions = predictions


#----------------------------------
# 5. Review Page
#----------------------------------


# Desgin Review Page
elif page == "review":
    st.header("🔍 Suspicious Account Review")

    # Set up session state for page history
    if "suspicious_accounts" not in st.session_state:

        # Retrieve data from session state
        df = st.session_state.get("df", pd.DataFrame())
        graph = st.session_state.get("graph", None)
        accounts = st.session_state.get("accounts", [])
        embeddings = st.session_state.get("embeddings", pd.DataFrame())
        predictions = st.session_state.get("predictions", [])

        # Load all accounts
        suspicious_accounts = extract_suspicious_accounts(df, graph, accounts, embeddings, predictions)

        # Change column names for better readability
        suspicious_accounts.columns = ["Account", "Flag"]

        # Add empty columns that stores the decision
        suspicious_accounts["Decision"] = ""

    # Use updated version from the investigation page
    else:
        suspicious_accounts = st.session_state.suspicious_accounts


    # Check if there are any suspicious accounts
    if suspicious_accounts.empty:
        st.warning("No suspicious accounts found.")
        st.stop()
    
    # Display the selected account's details
    st.dataframe(suspicious_accounts)

    # Store in session state
    st.session_state.suspicious_accounts = suspicious_accounts

    # Back button aligned to the left and Start Investigation button aligned to the far right
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Go Back to Home"):
            st.session_state.page = "title"
            st.rerun()
    with col3:
        if st.button("Start Investigation"):
            st.session_state.page = "investigate"
            st.rerun()
    

    # Build update functions
    st.markdown(
        """
        ## Update
        Review flagged accounts and update the model.
        """
    )

    # Add an Update button
    if st.button("Update"):
        st.session_state.page = "update"
        st.rerun()

    # Build monitor functions
    st.markdown(
        """
        ## Monitor
        Monitor the model performance after updates.
        """
    )

    # Add a Monitor button
    if st.button("Monitor"):
        st.session_state.page = "monitor"
        st.rerun()



#----------------------------------
# 6. Investigation Page
#----------------------------------


# Design Investigation Page
elif page == "investigate":
    st.header("Investigation of Suspicious Account")

    # Set up session state for page history
    if "page_history" not in st.session_state:
        st.session_state.page_history = []

    # Retrive Suspicious Accounts from session state
    suspicious_accounts = st.session_state.get("suspicious_accounts", pd.DataFrame())

    # Retrieve DataFrame from session state
    df = st.session_state.get("df", pd.DataFrame())

    # Get list of accounts
    account_list = suspicious_accounts["Account"].tolist()

    # Initialize account index
    if "current_account_idx" not in st.session_state:
        st.session_state.current_account_idx = 0

    # Load the selected account
    selected_account = st.selectbox(
        "Select an Account to Investigate",
        account_list,
        index=st.session_state.current_account_idx,
        key="account_selector"
    )

    # Sync session index if user picks something new
    new_index = account_list.index(selected_account)
    if new_index != st.session_state.current_account_idx:
        st.session_state.current_account_idx = new_index
        st.rerun()

    # Automatically get account from current index
    if st.session_state.current_account_idx >= len(suspicious_accounts):
        st.success("✅ All accounts have been reviewed!")
        if st.button("Return to Review Page"):
            st.session_state.page = "review"
            st.rerun()
        st.stop()

    selected_account = suspicious_accounts["Account"].iloc[st.session_state.current_account_idx]
    st.markdown(f"### Currently Investigating: `{selected_account}`")

    # Call function to get the transactions and graph
    node_features, sent_transactions, received_transactions, G =  create_investigate_elements(suspicious_accounts, selected_account, df)


    # Display the features without the index
    st.subheader(f"Node Features for {selected_account}")
    st.dataframe(node_features, use_container_width=True, hide_index=True)


    # Display the transactions without the index
    st.subheader("Transactions Sent by the Account")
    st.dataframe(sent_transactions, use_container_width=True, hide_index=True)

    # Display the transactions without the index
    st.subheader("Transactions Received by the Account")
    st.dataframe(received_transactions, use_container_width=True, hide_index=True)


    # Define position for the graph
    pos = nx.spring_layout(G, seed=187)

    # Set node colors: red for the selected account, blue for others
    node_colors = ["red" if node == selected_account else "lightblue" for node in G.nodes()]

    plt.figure(figsize=(10, 6))
    nx.draw(
        G, pos, with_labels=True, node_size=1000, 
        node_color=node_colors, font_size=10, 
        font_color="black", edge_color="gray"
    )
    plt.title(f"Transaction Network for {selected_account}")
    st.subheader("Transaction Network")
    st.pyplot(plt)
    plt.clf()

    # Decision buttons
    col1, col2, col3 = st.columns([1, 4, 1])
    if col1.button("✅ Not AML", key="not_fraud"):
        st.session_state.suspicious_accounts.loc[
            st.session_state.suspicious_accounts["Account"] == selected_account, "Decision"
        ] = "Not Fraud"
        #st.success(f"{selected_account} marked as NOT fraud.")
        st.session_state.current_account_idx += 1
        st.rerun()
    if col3.button("🚨🚨 AML", key="fraud"):
        st.session_state.suspicious_accounts.loc[
            st.session_state.suspicious_accounts["Account"] == selected_account, "Decision"
        ] = "Fraud"
        #st.error(f"{selected_account} marked as FRAUD.")
        st.session_state.current_account_idx += 1
        st.rerun()


    # Create buttons to move forawrd and backward through the accounts
    col1, col2, col3 = st.columns([1, 4, 1])
    if col1.button("⬅️ Prev. Account", key="previous_account"):
        # Decrement the index if it's greater than 0
        if st.session_state.current_account_idx > 0:
            st.session_state.current_account_idx -= 1
        st.rerun()
    if col3.button("➡️ Next Account", key="next_account"):
        # Increment the index if it's less than the length of suspicious accounts
        if st.session_state.current_account_idx < len(suspicious_accounts) - 1:
            st.session_state.current_account_idx += 1
        st.rerun()

    # Back button with rerun-safe logic
    if st.button("⬅️ Back to Review Page", key="back_button"):
        st.session_state.page = "review"
        st.rerun()

#----------------------------------
# 7. Update Page
#----------------------------------
    
# Design Update Page
elif page == "update":

    # Set header
    st.header("Update Model and Report Files")

    # Retrive Suspicious Accounts from session state
    suspicious_accounts = st.session_state.get("suspicious_accounts", pd.DataFrame())

    # Filter all investigated accounts
    investigated_accounts = suspicious_accounts[suspicious_accounts["Decision"] != ""]

    # Set up header
    st.subheader("Summary of Decisions")

    # Check if there are any investigated accounts
    if investigated_accounts.empty:
        st.warning("No accounts have been investigated yet.")
        #st.stop()

    else:
        st.dataframe(investigated_accounts, use_container_width=True, hide_index=True)

    
        # Set up button to update model
        if st.button("Update Model"):
            
            # Retrieve embeddings
            embeddings = st.session_state.get("embeddings", pd.DataFrame())

            # Run Update Model function
            results = update_model(embeddings, investigated_accounts)
            
            # Remove investigated accounts from suspicious accounts
            st.session_state.suspicious_accounts = st.session_state.suspicious_accounts[
                st.session_state.suspicious_accounts["Decision"] == ""
            ]

            # Save the updated suspicious accounts to session state
            st.session_state.suspicious_accounts.reset_index(drop=True, inplace=True)

            # Save the results to the session state
            st.session_state.update_results = results

            # Reset Index
            st.session_state.current_account_idx = 0

            st.success("Model updated successfully!")

            # Automatically redirect back to the review page
            st.session_state.page = "review"
            st.rerun()

    
    # Back button with rerun-safe logic
    if st.button("⬅️ Back to Review Page", key="back_button"):
        st.session_state.page = "review"
        st.rerun()

#----------------------------------
# 7. Monitor Page
#----------------------------------


# Create a page to monitor the model performance
elif page == "monitor":

    st.header("Model Performance Monitoring")

    # Retrieve the update results from session state
    if "update_results" in st.session_state:

        update_results = st.session_state.get("update_results", None)

        # Apply function to get monitoring outputs:
        correct, incorrect, overall_results = monitor_items(update_results)


        # Show the results from the last update
        st.subheader("Last Update Results")

        st.markdown(
            """
            The table below shows the **correctly** identified predictions by the model from the last update. 
            """
        )

        st.dataframe(correct, use_container_width=True, hide_index=True)

        st.markdown(
            """
            The table below shows the **incorrectly** identified predictions by the model from the last update. 
            """
        )
        st.dataframe(incorrect, use_container_width=True, hide_index=True)

        # Show the average number of corrects
        st.markdown("### Summary Statistics of Last Update")
        st.write(f"**{len(correct)}** out of **{len(update_results)}** predictions were correct.")
        st.write(f"Number of False Positives: **{len(incorrect)}**")

    else:
        # Show the results from the last update
        st.subheader("Last Update Results")
        

        # Print a message if no updates have been made
        st.warning("No updates have been made yet. Please update the model first.")

        # Load the dataframe wiithout including the new update
        overall_results = get_overall_results()


    # Show the overall results
    st.subheader("Overall Results")

    # Get Total accuaracy
    accuracy = (overall_results["Actual Label"] == 1).sum() / len(overall_results)
    st.write(f"**Total Accuracy:** {accuracy:.2f}")

    # Get Total False Positives
    fp = (overall_results["Actual Label"] == 0).sum() / len(overall_results)
    st.write(f"**Total False Positive Rate:** {fp:.2f}")

    # Plot accuary and false positive rate per run
    st.markdown("**Model Performance Over Time**")

    # Retrive items for the plot
    runs, accuracy_per_run, false_positive_rate_per_run = monitor_plot(overall_results)

    # Plot the metrics
    plt.figure(figsize=(10, 6))
    plt.plot(runs, accuracy_per_run, label="Accuracy", marker="o")
    plt.plot(runs, false_positive_rate_per_run, label="False Positive Rate", marker="o", linestyle="--")
    plt.xlabel("Run")
    plt.ylabel("Metric Value")
    plt.title("Model Performance Over Time")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)
    plt.clf()

    st.markdown("**Summary Dataframe with all Investigated Accounts**")
    st.dataframe(overall_results, 
            use_container_width=True, 
            hide_index=True
    )

    # Back button with rerun-safe logic
    if st.button("⬅️ Back to Review Page", key="back_button"):
        st.session_state.page = "review"
        st.rerun()