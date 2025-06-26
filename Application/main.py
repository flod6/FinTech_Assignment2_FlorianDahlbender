#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import os
from pathlib import Path
import numpy as np

# Set paths
dir = Path(__file__).resolve().parent.parent

# Define app path
app_path = dir / "Application" / "app.py"
app_path_str = str(app_path)

os.system(f'STREAMLIT_EMAIL_PROMPT=false streamlit run "{app_path_str}"')

# Set Seed
np.random.seed(6)


