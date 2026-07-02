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

The dataset used in this project is stored locally at `data/diabetes.csv`.

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

The main Python script is `main.py`.

## Results

With the current dataset and random state, the script produces an accuracy of `0.7239583333333334` and `25` false negatives.

## Understanding the Confusion Matrix

The confusion matrix compares correct and incorrect predictions for both diabetes and non-diabetes outcomes.

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

From the project folder, run `python main.py`.

If you are using the local virtual environment created for this project, run `./.venv/bin/python main.py`.

## Project Structure

The project contains the dataset in `data/diabetes.csv`, the main script in `main.py`, and project configuration files such as `pyproject.toml` and `uv.lock`.

## Conclusion

This project demonstrates how Logistic Regression can be used for binary medical classification. The model predicts whether a patient has diabetes based on diagnostic measurements, then evaluates its performance using accuracy and a confusion matrix.

The most important part of the evaluation is understanding the real-world consequences of incorrect predictions. In this diabetes prediction task, false negatives are especially serious because they may delay diagnosis and treatment.
