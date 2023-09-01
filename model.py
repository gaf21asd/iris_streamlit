# -*- coding:utf-8 -*-
import pandas as pd 

# Label encoding
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split # split the dataset
from sklearn.linear_model import LogisticRegression # Linear model - Logistic Regression
import joblib # Library for saving the model

# Update the model
data = pd.read_csv("data/iris.csv")

# Applying label encoding to dependent features.
le = LabelBinarizer()
# print(le.fit(data['species']))
data['species'] = le.fit_transform(data['species'])
# print(le.classes_)
# print(data['species'])

# Separate independent, dependent features.
X = data.drop(columns=['species'])
y = data['species']

# Validation phase for training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

model = LogisticRegression()
model.fit(X_train, y_train) # Modeling code for training X and y

model_file = open("models/lgr_model_iris0901.pkl", "wb")
joblib.dump(model, model_file) # Export
model_file.close() 