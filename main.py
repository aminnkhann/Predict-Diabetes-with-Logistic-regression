import pandas as pd  # Import pandas so we can load and work with the CSV data as a DataFrame.
from pathlib import Path  # Import Path so the file path works reliably from this project folder.
from sklearn.model_selection import train_test_split  # Import the tool that splits the data into train and test parts.
from sklearn.linear_model import LogisticRegression  # Import the Logistic Regression machine learning model.
from sklearn.metrics import accuracy_score, confusion_matrix  # Import the tools used to evaluate the model.


# Create the path to the folder where the dataset is stored.
DATA_DIR = Path(__file__).parent.resolve() / "data"

# Load the diabetes.csv file into a pandas DataFrame.
diabetes_data = pd.read_csv(DATA_DIR / "diabetes.csv")

# Create X, which contains only the input features used to make predictions.
X = diabetes_data.drop("Outcome", axis=1)

# Create y, which contains the target answer that the model should learn to predict.
y = diabetes_data["Outcome"]

# Split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,  # The medical measurement columns used as input data.
    y,  # The real diabetes result for each patient.
    test_size=0.25,  # Use 25% of the data for testing and 75% for training.
    random_state=42  # Keep the split the same every time the script runs.
)

# Create the Logistic Regression model.
model = LogisticRegression(max_iter=1000)

# Train the model using the training data.
model.fit(X_train, y_train)

# Use the trained model to predict diabetes results for the testing data.
y_pred = model.predict(X_test)

# Calculate how often the model predicted the correct answer.
accuracy = accuracy_score(y_test, y_pred)

# Create a confusion matrix to compare real answers with predicted answers.
cm = confusion_matrix(y_test, y_pred)

# Get the number of false negatives from the confusion matrix.
false_negatives = cm[1][0]

# Print the accuracy score.
print("Accuracy:", accuracy)

# Print a label before showing the confusion matrix.
print("Confusion Matrix:")

# Print the confusion matrix.
print(cm)

# Print the number of false negatives.
print("False Negatives:", false_negatives)


# Explanation:
# In this assignment, we used Logistic Regression to predict whether a patient has diabetes.
# First, we loaded the diabetes.csv file into a pandas DataFrame.
# Then we separated the dataset into X and y.
# X means the features, which are the medical measurements such as Glucose, BloodPressure, BMI, Age, and Insulin.
# y means the target, which is the real answer we want the model to learn: Outcome.
# In this dataset, Outcome = 0 means the patient does not have diabetes, and Outcome = 1 means the patient has diabetes.
# After that, we split the data into training data and testing data.
# The training data is used to teach the model.
# The testing data is used to check how well the model works on patients it has not seen before.
# We used 75% of the data for training and 25% of the data for testing.
# Logistic Regression is a classification model, which means it predicts categories.
# In this case, the categories are "diabetes" or "no diabetes".
# The accuracy score shows the percentage of correct predictions.
# The confusion matrix shows more detail about the model's correct and incorrect predictions.
# The confusion matrix is arranged like this:
# [[True Negative, False Positive],
#  [False Negative, True Positive]]
# True Negative means the patient did not have diabetes, and the model correctly predicted no diabetes.
# False Positive means the patient did not have diabetes, but the model predicted diabetes.
# False Negative means the patient did have diabetes, but the model predicted no diabetes.
# True Positive means the patient did have diabetes, and the model correctly predicted diabetes.
# In my result, the model produced 25 false negatives.
# A false negative is dangerous because a patient who actually has diabetes might not get treatment or lifestyle advice.
# This could allow the disease to get worse and cause serious health problems later.
# A false positive can also be stressful because a patient may worry and need extra tests.
# However, in real life, a false positive is usually less dangerous than a false negative because more testing can correct it.
