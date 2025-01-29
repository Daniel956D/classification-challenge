# classification-challenge

Spam Detection Model Comparison
Project Overview
This project implements and compares two machine learning models (Logistic Regression and Random Forest) for spam detection. The models are trained on a dataset containing various features extracted from messages to classify them as either spam or legitimate.
Features

Data preprocessing and feature scaling
Implementation of two classification models:

Logistic Regression
Random Forest Classifier


Model performance comparison and evaluation
Detailed accuracy metrics

Technologies Used

Python 3.x
pandas
scikit-learn
numpy

Installation

Clone the repository:

bashCopygit clone [your-repository-url]

Install required packages:

bashCopypip install pandas numpy scikit-learn
Dataset
The dataset used in this project contains message features and their corresponding labels (spam/not spam). The features include various characteristics extracted from messages, and the target variable is binary (0 for legitimate messages, 1 for spam).
Source: https://static.bc-edx.com/ai/ail-v-1-0/m13/challenge/spam-data.csv
Implementation Details
Data Preprocessing

Split data into features (X) and labels (y)
Check class balance using value_counts
Split data into training and testing sets
Scale features using StandardScaler

Model Training
Two models were implemented and compared:

Logistic Regression

Basic parameters with random_state=1
Trained on scaled data


Random Forest Classifier

Basic parameters with random_state=1
Trained on scaled data



Results

Logistic Regression Accuracy: 92.01%
Random Forest Accuracy: 95.92%

The Random Forest model outperformed the Logistic Regression model by approximately 3.91 percentage points, demonstrating superior performance in spam detection.
Code Sources and References
Libraries and Documentation

Scikit-learn Documentation: https://scikit-learn.org/stable/

train_test_split: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html


Pandas Documentation: https://pandas.pydata.org/docs/
NumPy Documentation: https://numpy.org/doc/

Implementation References
The code implementation follows standard machine learning practices and utilizes common patterns from:

Scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
Scikit-learn Tutorials: https://scikit-learn.org/stable/tutorial/index.html

Usage
To run the models:
pythonCopy# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess data
data = pd.read_csv("spam-data.csv")
X = data.drop("spam", axis=1)
y = data["spam"]

# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate models
# [Rest of the implementation as shown in the notebook]
Future Improvements

Feature importance analysis
Hyperparameter tuning
Cross-validation implementation
Additional evaluation metrics (precision, recall, F1-score)
Testing with different algorithms

Author
Daniel Dominugez 
