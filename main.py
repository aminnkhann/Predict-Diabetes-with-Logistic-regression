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
# Create a confusion matrix to compare real answers with predicted answers.
def confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm

# Get the number of false negatives from the confusion matrix.
def false_negatives(cm):
    return cm[1][0]

def print_results(accuracy, cm, false_negatives):
    # Print the accuracy score.
    print("Accuracy:", accuracy)

    # Print a label before showing the confusion matrix.
    print("Confusion Matrix:")

    # Print the confusion matrix.
    print(cm)

    # Print the number of false negatives.
    print("False Negatives:", false_negatives)


if __name__ == '__main__':
    diabetes_data = load_data()
    X, y = features_and_target(diabetes_data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = create_model()
    train_model(model, X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = evaluate_model(y_test, y_pred)
    cm = confusion(y_test, y_pred)
    fn = false_negatives(cm)
    print_results(accuracy, cm, fn)