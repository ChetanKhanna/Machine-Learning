# importing modules
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

# Data visualization
# loading dataframes
df = pd.read_csv('./iris.data')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1:].values
# preprocessing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# conveting 4D dataset to 2D for visualization
pca = PCA(n_components=2)
principal_components = pca.fit(X_test)
X_test_pca = pca.transform(X_test)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_)

# Fastening ML algo
# loading df
df_train = pd.read_csv('./mnist_train.csv')
df_test = pd.read_csv('./mnist_test.csv')
X_train = df_train.drop('label', axis=1).values
y_train = df_train['label'].values
X_test = df_test.drop('label', axis=1).values
y_test = df_test['label'].values
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# training and fitting pca
pca = PCA(0.95)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
regressor = LogisticRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(regressor.score(X_test, y_test))
print(pca.n_components_)
