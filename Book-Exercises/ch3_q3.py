# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
# importing sklearn packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


train_filename = os.path.join(os.path.expanduser('~'),
                              'Machine Learning/titanic_train.csv')
test_filename = os.path.join(os.path.expanduser('~'),
                             'Machine Learning/titanic_test.csv')
train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)
# print(train_df.head())
# print(train_df.info())
# 'Age', 'Cabin', 'Embarked' have Nans
train_df['Age'].fillna((train_df['Age'].median()), inplace=True)
train_df['Embarked'].fillna(random.choice(['S', 'Q', 'C']), inplace=True)
train_df.drop(['Cabin', 'Name'], axis=1, inplace=True)
# print(train_df.info())
# print(test_df.info())
test_df['Age'].fillna((test_df['Age'].median()), inplace=True)
test_df['Embarked'].fillna(random.choice(['S', 'Q', 'C']), inplace=True)
test_df['Fare'].fillna((test_df['Fare'].median()), inplace=True)
test_df.drop(['Cabin', 'Name'], axis=1, inplace=True)
# print(test_df.info())
# Making features and labels
y_train = train_df['Survived'].values
X_train_df = train_df.drop(['PassengerId', 'Survived'], axis=1)
# print(X_train[0:5, :])
# print(y_train[0:5])
# print(X_train.shape)
X_test_df = test_df.drop(['PassengerId'], axis=1)
# Combining df for Encoder and Scaler
X_combined_df = pd.concat([X_train_df, X_test_df])
# Encoder
enc = OneHotEncoder(sparse=False)
X_combined = enc.fit_transform(X_combined_df)
# Scaler
scaler = StandardScaler()
X_combined = scaler.fit_transform(X_combined)
X_train, X_test = train_test_split(X_combined, train_size=len(y_train),
                                   random_state=42)
# training model
model = KNeighborsClassifier()
print("Training model..")
model.fit(X_train, y_train)
print('Done.')
print('Getting accuracy')
y_pred = model.predict(X_test)
print(cross_val_score(model, X_test, y_pred, cv=3))
