from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

import matplotlib.pyplot as plt


df = fetch_mldata('MNIST original')
X, y = df['data'], df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# BINARY CLASSIFIACTION
# Making targets for classifying digit 2
y_train_2 = (y_train == 2)
y_test_2 = (y_test == 2)
# Seperating an example
some_digit_index = 100
some_digit = X_test[some_digit_index]
# Creating and fitting the model
model = SGDClassifier()
model.fit(X_train, y_train_2)
# print(model.fit(X_test)) # prints model prediction
# Calculating cross validation scores
score_list = cross_val_score(model, X_test, y_test_2, cv=4,
                             scoring='accurancy')
# print(score_list)
# Splitting into tets and train predictions --
# Just for a better undertanding of when to use test set
y_pred_from_train = model.predict(X_train)
y_pred_from_test = model.predict(X_test)
# Getting the confusion metrix and the f1 score
cm_train = confusion_matrix(y_train_2, y_pred_from_train)
cm_test = confusion_matrix(y_test_2, y_pred_from_test)
print('traim:\n', cm_train, '\n',
      'test:\n', cm_test)
print('train: ', f1_score(y_train_2, y_pred_from_train),
      'test:', f1_score(y_test_2, y_pred_from_test))
# Precisoin-Recall tradeoff -- deciding custom threshold
y_score = cross_val_predict(model, X_train, y_train_2, cv=3,
                            method='decision_function')
precisions, recalls, threshholds = precision_recall_curve(
    y_train_2, y_score)
plt.plot(threshholds, precisions[:-1], 'b--', label='precision')
plt.plot(threshholds, recalls[:-1], 'g-', label='recall')
plt.xlabel('Thrreshhold')
plt.ylim([0, 1])
plt.legend()
plt.show()
# Say we want a precisino of 90% then look at the above curve and find
# a suitbale threshold for 90% precision. Let's say the value is 70000
y_pred_2_threshhold = (y_score > 70000)

# Multi-class classification
# By defualt, most binary classfiers when given a multi-class
# problem, use OvA strategy, except a few like SVMs which use OVO
# as it is faster. We can also explicitly specify to Sklearn which
# strategy to use.
multi_model = SGDClassifier()
multi_model.fit(X_train, y_train)  # this will use OvA
# In reality, this will use 10 different Binary classifiers
multi_scores = multi_model.decision_function([some_digit])
# This will store an array of 10 scores in scores_multi
# The one with the highest index will be selected by predict_ method
# To explicityly use specified methiods
OvO_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
OvO_clf.fit(X_train, y_train)

# Random-Forest Classifer
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)
rf_clf.predict_proba([some_digit])
y_pred_multi = cross_val_predict(multi_model, X_train, y_train, cv=3)
conf_mat = confusion_matrix(y_train, y_pred_multi)
plt.plot(conf_mat, cmap=plt.cm.gray)
# plt.show()
# The row represents the actual classes while the col
# represents the predicted classes.

# Multi-Label Classification -- classifying multiple
# binary labels on one input
# Creating a new label array
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_train_multi = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train_multi)
knn_clf.predict([some_digit])
# Evaluating a multi-label classifier
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv=3)
score = f1_score(y_train, y_train_knn_pred, average='macro')
