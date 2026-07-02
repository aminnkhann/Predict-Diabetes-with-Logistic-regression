import pandas as pd  # Import pandas so we can load and work with the CSV data as a DataFrame.
from pathlib import Path  # Import Path so the file path works reliably from this project folder.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Create the path to the folder where the dataset is stored.
DATA_DIR = Path(__file__).parent.resolve() / "data"

# Load the diabetes.csv file into a pandas DataFrame.
def load_data():
    return pd.read_csv(DATA_DIR / "diabetes.csv")

# Create X, which contains only the input features used to make predictions.
# Create y, which contains the target answer that the model should learn to predict.
def features_and_target(df):
    return df.drop("Outcome", axis=1), df["Outcome"]

# Split the data into training and testing sets.
def split_data(X, y):
        return train_test_split(X, y, test_size=0.25, random_state=42)
# Create a Logistic Regression model.
def create_model():
    return LogisticRegression(max_iter=1000)

# Train the model.
def train_model(model, X, y):
    model.fit(X, y)

# Use the trained model to predict diabetes results for the testing data.
def predict(model, X):
    return model.predict(X)

# Calculate how often the model predicted the correct answer.
def evaluate_model(y_test, y_pred):
     accuracy = accuracy_score(y_test, y_pred)

     return accuracy
