import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# making the datasets
df = pd.read_csv('./spam.csv', encoding='latin-1')
print(df.info())
print(df.head())
encoder_1 = LabelEncoder()
y = encoder_1.fit_transform(df['v1'])
encoder_2 = OneHotEncoder(sparse=False)
X = encoder_2.fit_transform(df['v2'].values.reshape(-1, 1))
print(X.shape, y.shape)
# applying PCA
pca = PCA(n_components=0.95) # use svd_solver='randomized'
							 # when n_features <<< n_instances
							 # to use a stochasitc algorithm
X_pca = pca.fit_transform(X)
print(X_pca.shape)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y,
                                                    test_size=0.3,
                                                    random_state=42)
# applying SVM classifier on reduced features
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
y_pred = svm_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# Imcremental PCA for large datasets
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
# PCA
inc_pca = IncrementalPCA(n_components=160)
n_batch = 100
for X_batch in np.array_split(X_train, n_batch):
	inc_pca.partial_fit(X_batch)
X_pca = inc_pca.transform(X_train)
print(X_pca.shape)
