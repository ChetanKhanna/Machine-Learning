# import required packages
import pandas as pd
import os
# import sklearn packages
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

train_filepath = os.path.join(os.path.expanduser('~'), 'Machine Learning',
                              'mnist_train.csv')
test_filepath = os.path.join(os.path.expanduser('~'), 'Machine Learning',
                             'mnist_test.csv')
df_train = pd.read_csv(train_filepath)
df_test = pd.read_csv(test_filepath)
print(df_train.head())
print(df_test.head())
X_train = df_train.iloc[:, 1:].astype('int64')
y_train = df_train.iloc[:, 0].astype('int64')
X_test = df_test.iloc[:, 1:].astype('int64')
y_test = df_test.iloc[:, 0].astype('int64')
# model training
model = KNeighborsClassifier()
model.fit(X_train, y_train)
print(cross_val_score(model, X_test, y_test, cv=3))
# Hyper-paramter tuning
n_neighbors = [3, 4, 5, ]
weights = ['uniform', 'distance']
hyperparams = dict(n_neighbors=n_neighbors, weights=weights)
search = GridSearchCV(model, hyperparams, cv=3, n_jobs=-1)
search.fit(X_test, y_test)
print('best-score: ', search.best_score_)
print('best-params: ', search.best_params_)
