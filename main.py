import pandas as pd  # Import pandas so we can load and work with the CSV data as a DataFrame.
from pathlib import Path  # Import Path so the file path works reliably from this project folder.


# Create the path to the folder where the dataset is stored.
DATA_DIR = Path(__file__).parent.resolve() / "data"

# Load the diabetes.csv file into a pandas DataFrame.
def load_data():
    return pd.read_csv(DATA_DIR / "diabetes.csv")

