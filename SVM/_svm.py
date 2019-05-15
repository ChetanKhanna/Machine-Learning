from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import datasets
import pandas as pd
import os

# SVM on Linearly-sperable data
# loading DB
bankdata = pd.read_csv('./bill_authentication.csv')
# data analysis
print(bankdata.shape)
print(bankdata.head())
_ = input('press key to continue..')
os.system('clear')
# data preprocessing
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=22)
# trainig classifer
clf = svm.SVC(kernel='linear')
clf.fit(X_test, y_test)
y_pred = clf.predict(X_test)
# scoring using confusion metric
print('Linear Data')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SVM on non-linear data -- Kernals
# polynomial kernal
irisdata = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    irisdata.data, irisdata.target, test_size=0.2, random_state=22)
clf = svm.SVC(kernel='poly', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('polynomial kernal')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# gaussian kernal
clf = svm.SVC(kernel='rbf', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('gaussian kernal')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# sigmoid kernal
clf = svm.SVC(kernel='sigmoid', gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print('sigmoid kernal')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
# NOTE: sigmoid kernal is usually used for binary classification
# and herce is likely to underperform since we have three clf classes
