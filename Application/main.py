#----------------------------------
# 1. Set Up
#----------------------------------

# Load Packages
import os
from pathlib import Path

# Set paths
dir = Path(__file__).resolve().parent.parent

# Define app path
app_path = dir / "Application" / "app.py"
app_path_str = str(app_path)

#----------------------------------
# 2. Run App
#----------------------------------

os.system(f'STREAMLIT_EMAIL_PROMPT=false streamlit run "{app_path_str}"')