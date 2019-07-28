# importing required modules
import pandas as pd
import os
# importing sklearn packages
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

train_filename = os.path.join(os.path.expanduser('~'),
                              'Machine Learning', 'mnist_train.csv')
test_filename = os.path.join(os.path.expanduser('~'),
                             'Machine Learning', 'mnist_test.csv')
train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)
print(train_df.info())
print(test_df.info())
X_train = train_df.iloc[:, 1:].astype('float64')
y_train = train_df.iloc[:, 0].astype('float64')
X_test = test_df.iloc[:, 1:].astype('float64')
y_test = test_df.iloc[:, 0].astype('float64')
# scaling dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# training SVM
model = LinearSVC(multi_class='ovr', max_iter=5000, C=0.1)
print('Training model..')
model.fit(X_train, y_train)
print('Done.')
print('Calculating accuracy..')
print(cross_val_score(model, X_test, y_test, cv=3))
