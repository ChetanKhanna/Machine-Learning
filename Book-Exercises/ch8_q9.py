# importing packages
import pandas as pd
import os
from datetime import datetime
# importing sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score


train_filename = os.path.join(os.path.expanduser('~'),
                              'Machine Learning', 'mnist_train.csv')
test_filename = os.path.join(os.path.expanduser('~'),
                             'Machine Learning', 'mnist_test.csv')
train_df = pd.read_csv(train_filename)
test_df = pd.read_csv(test_filename)
X_train = train_df.iloc[:, 1:].astype('float64')
y_train = train_df.iloc[:, 0].astype('float64')
X_test = test_df.iloc[:, 1:].astype('float64')
y_test = test_df.iloc[:, 0].astype('float64')
# scaling dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# training Random Forest without PCA
print(X_train.shape, X_test.shape)
rf_clf = RandomForestClassifier(n_estimators=5)
lr_clf = LogisticRegression()
print('Running classifer..')
start_time = datetime.now()
lr_clf.fit(X_train, y_train)
end_time = datetime.now()
print('Without PCA:\nTime-taken:', end_time - start_time, '\nAccuracy:',
      accuracy_score(y_test, lr_clf.predict(X_test)))
print('Applying PCA..')
pca = PCA(n_components=0.95)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
print('Done.')
print(X_train.shape, X_test.shape)
lr_clf_2 = RandomForestClassifier(n_estimators=5)
print('Running classifer..')
start_time_2 = datetime.now()
lr_clf_2.fit(X_train, y_train)
end_time_2 = datetime.now()
print('With PCA:\nTime-taken:', end_time_2 - start_time_2, '\nAccuracy:',
      accuracy_score(y_test, lr_clf_2.predict(X_test)))
# Note : PCA does not reduce the training time always; here the training time 
#        increases and also the drop in accuracy is significant when using
#        RandomForest. On the other hand, using LogisticRegression significantly
#        reduces the training time and the decline in performance is also acceptable
