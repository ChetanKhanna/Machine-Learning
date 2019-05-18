# importing modules
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import time


# Data visualization
# loading dataframes
df = pd.read_csv('./iris.data')
X = df.iloc[:, :-1].astype('float64')
y = df.iloc[:, -1:].values
# preprocessing data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# conveting 4D dataset to 2D for visualization
pca = PCA(n_components=2)
principal_components = pca.fit(X_train)
X_train_pca = pca.transform(X_train)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_)
# Plotting scatter plots for the two features
# Building Dataframe using pandas
component_df = pd.DataFrame(data=X_train_pca,
                            columns=['component 1', 'component 2'])
label_df = pd.DataFrame(data=y_train, columns=['label'])
# plotting a scatter plot
indices_to_keep = label_df['label'] == 'Iris-setosa'
plt.scatter(component_df.loc[indices_to_keep, 'component 1'],
            component_df.loc[indices_to_keep, 'component 2'],
            c='r', s=10)
indices_to_keep = label_df['label'] == 'Iris-versicolor'
plt.scatter(component_df.loc[indices_to_keep, 'component 1'],
            component_df.loc[indices_to_keep, 'component 2'],
            c='b', s=10)
indices_to_keep = label_df['label'] == 'Iris-virginica'
plt.scatter(component_df.loc[indices_to_keep, 'component 1'],
            component_df.loc[indices_to_keep, 'component 2'],
            c='g', s=10)
plt.show()
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
# pca = PCA(0.95)
# pca.fit(X_train)
# X_train = pca.transform(X_train)
# X_test = pca.transform(X_test)
# regressor = LogisticRegression()
# regressor.fit(X_train, y_train)
# y_pred = regressor.predict(X_test)
# print(regressor.score(X_test, y_test))
# print(pca.n_components_)
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))
# comparing PCA for various number of components
for variance in [1.0, 0.99, 0.95, 0.90, 0.85]:
    start_time = time.time()
    pca = PCA(0.95)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    regressor = LogisticRegression()
    regressor.fit(X_train, y_train)
    end_time = time.time()
    y_pred = regressor.predict(X_test)
    print('variance retained:', variance)
    print('number of components:', pca.n_components_)
    print('time taken:', end_time - start_time)
    print('accuracy:', regressor.score(X_test, y_test))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
