# importing packages
import pandas as pd
import os
import numpy as np
# importing sklearn packages
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
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
X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=1/6,
                                                random_state=42)
# training model
rf_clf = RandomForestClassifier()
et_clf = ExtraTreesClassifier()
svm_clf = SVC()
kn_clf = KNeighborsClassifier()
lr_clf = LogisticRegression()
vt_clf = VotingClassifier(estimators=[('rf', rf_clf), ('et', et_clf),
                          ('svm', svm_clf), ('kn', kn_clf), ('lr', lr_clf)],
                          voting='hard')
print('Running models and predictions..')
# printing individual score
for clf in (lr_clf, svm_clf, rf_clf, et_clf, kn_clf, vt_clf):
    print('Training', clf.__class__.__name__, '...')
    clf.fit(X_train, y_train)
    print('Predicting labels from model')
    y_pred = clf.predict(X_cv)
    print('Getting accuracy')
    print(accuracy_score(y_cv, y_pred))
# Creating a blender using RandomForestClassifier as predictor
estimators = [rf_clf, et_clf, svm_clf, kn_clf, lr_clf]
# making the new X_pred which contains predictions of all estimators
X_pred = np.empty((len(X_cv), len(estimators)), dtype=np.float32)
for index, estimator in enumerate(estimators):
    X_pred[:, index] = estimator.predict(X_cv)
# training blender
rf_blender = RandomForestClassifier(n_estimators=200, oob_score=True,
                                    random_state=42)
rf_blender.fit(X_pred, y_cv)
print('Acrruacy for blender:', rf_blender.oob_score_)