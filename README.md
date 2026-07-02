# Diabetes Prediction with Logistic Regression

## Project Overview

This project applies **Logistic Regression** to a real-world medical dataset in order to predict whether a patient is likely to have diabetes based on diagnostic measurements.

The assignment is based on the same classification workflow used in class with the Breast Cancer dataset, but here the model is trained on a diabetes dataset instead.

The goal is not only to build a working machine learning model, but also to understand how the model's results should be interpreted in a real healthcare context.

## Project Goals

The main goals of this project are:

- Load a diabetes dataset from a CSV file into a pandas DataFrame.
- Separate the dataset into input features and a target variable.
- Split the data into training and testing sets.
- Train a Logistic Regression model using scikit-learn.
- Make predictions on unseen test data.
- Evaluate the model using accuracy and a confusion matrix.
- Explain the real-world meaning of false positives and false negatives.

## Dataset

The dataset used in this project is stored locally in:

```text
data/diabetes.csv
```

The CSV file contains medical predictor variables and one target column.

### Expected Columns

| Column | Meaning |
| --- | --- |
| `Pregnancies` | Number of times the patient has been pregnant |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin fold thickness |
| `Insulin` | Serum insulin level |
| `BMI` | Body Mass Index |
| `DiabetesPedigreeFunction` | Diabetes heredity/pedigree score |
| `Age` | Patient age |
| `Outcome` | Target value: `0` means no diabetes, `1` means diabetes |

## Dataset Sources

If the CSV file is not already available locally, a compatible version of the dataset can commonly be found from these sources:

- [Kaggle: Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [UCI Machine Learning Repository: Diabetes dataset resources](https://archive.ics.uci.edu/dataset/34/diabetes)

When downloading from any external source, make sure the file matches the columns listed above. Some diabetes datasets use different formats, so the script expects a CSV with the target column named `Outcome`.

## Tools and Libraries

This project uses:

- Python
- pandas
- scikit-learn

The main Python script is:

```text
main.py
```

## Methodology

### 1. Load the Data

The dataset is loaded using pandas:

```python
diabetes_data = pd.read_csv(DATA_DIR / "diabetes.csv")
```

This converts the CSV file into a DataFrame, which makes it easier to inspect, select, and process columns.

### 2. Separate Features and Target

The model needs two parts:

- `X`: the input features used to make predictions
- `y`: the correct answer the model should learn to predict

In this project:

```python
X = diabetes_data.drop("Outcome", axis=1)
y = diabetes_data["Outcome"]
```

`X` contains the medical measurements, while `y` contains whether the patient has diabetes.

### 3. Split the Dataset

The data is split into:

- 75% training data
- 25% testing data

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)
```

The training set is used to teach the model. The testing set is used to check whether the model can make good predictions on data it has not seen before.

### 4. Train the Logistic Regression Model

The Logistic Regression model is initialized and trained:

```python
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

`max_iter=1000` gives the model enough iterations to finish training without convergence warnings.

### 5. Make Predictions

After training, the model predicts the diabetes outcome for the testing data:

```python
y_pred = model.predict(X_test)
```

These predictions are then compared with the real test answers.

### 6. Evaluate the Model

The model is evaluated using:

- Accuracy score
- Confusion matrix
- False negative count

```python
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
false_negatives = cm[1][0]
```

## Results

With the current dataset and random state, the script produces:

```text
Accuracy: 0.7239583333333334
Confusion Matrix:
[[95 28]
 [25 44]]
False Negatives: 25
```

## Understanding the Confusion Matrix

The confusion matrix is arranged like this:

```text
[[True Negative, False Positive],
 [False Negative, True Positive]]
```

For this project result:

| Metric | Value | Meaning |
| --- | ---: | --- |
| True Negative | 95 | Patient did not have diabetes, and the model predicted no diabetes |
| False Positive | 28 | Patient did not have diabetes, but the model predicted diabetes |
| False Negative | 25 | Patient had diabetes, but the model predicted no diabetes |
| True Positive | 44 | Patient had diabetes, and the model predicted diabetes |

## Real-World Interpretation

In a medical prediction problem, accuracy alone is not enough. The types of mistakes matter.

A **False Negative** means the patient actually has diabetes, but the model predicts that they do not. This is dangerous because the patient may not receive treatment, medical advice, or lifestyle changes early enough. In the real world, this could allow the disease to become worse and lead to serious complications.

A **False Positive** means the patient does not actually have diabetes, but the model predicts that they do. This can cause stress, unnecessary follow-up appointments, and extra medical testing. However, it is usually less dangerous than a false negative because additional medical tests can correct the mistake.

For this reason, in healthcare problems, reducing false negatives is often very important.

## How to Run the Project

From the project folder, run:

```bash
python main.py
```

If you are using the local virtual environment created for this project, run:

```bash
./.venv/bin/python main.py
```

## Project Structure

```text
diabetes/
├── data/
│   └── diabetes.csv
├── main.py
├── README.md
├── pyproject.toml
└── uv.lock
```

## Conclusion

This project demonstrates how Logistic Regression can be used for binary medical classification. The model predicts whether a patient has diabetes based on diagnostic measurements, then evaluates its performance using accuracy and a confusion matrix.

The most important part of the evaluation is understanding the real-world consequences of incorrect predictions. In this diabetes prediction task, false negatives are especially serious because they may delay diagnosis and treatment.
