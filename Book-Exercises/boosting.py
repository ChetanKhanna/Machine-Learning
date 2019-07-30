import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv('./spam.csv', encoding='latin-1')
print(df.info())
print(df.head())
encoder_1 = LabelEncoder()
y = encoder_1.fit_transform(df['v1'])
encoder_2 = OneHotEncoder(sparse=False)
X = encoder_2.fit_transform(df['v2'].values.reshape(-1, 1))
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
# # making Adaptive Boosting Classifer
# ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
#                              n_estimators=50, algorithm='SAMME.R',
#                              learning_rate=0.5) # if estimator doesn't have
#                                                 # predict_proba(), then use SAMME
# print('Training model..')
# ada_clf.fit(X_train, y_train)
# print('Done.')
# print('Making predictions..')
# y_pred = ada_clf.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, y_pred))
# Gradient Boosting -- Manual
# dt_clf_1 = DecisionTreeClassifier(max_depth=2)
# dt_clf_1.fit(X_train, y_train)
# y2 = y_train - dt_clf_1.predict(X_train)
# dt_clf_2 = DecisionTreeClassifier(max_depth=2)
# dt_clf_2.fit(X_train, y2)
# y3 = y2 - dt_clf_2.predict(X_train)
# dt_clf_3 = DecisionTreeClassifier(max_depth=2)
# dt_clf_3.fit(X_train, y3)
# # predicting using the three decision trees
# y_pred = sum(dt_clf.predict(X_test) for dt_clf in (dt_clf_1, dt_clf_2, dt_clf_3))
# print('Accuracy:', accuracy_score(y_test, y_pred))
# Gradient Boosting using package
# grd_clf = GradientBoostingClassifier(max_depth=2, n_estimators=3, learning_rate=1.0)
# print('Training Model..')
# grd_clf.fit(X_train, y_train)
# print('Done.')
# y_pred = grd_clf.predict(X_test)
# print('Accuracy:', accuracy_score(y_test, y_pred))
# Finding the optimal value for n_estimators
# Using a large arbit value and cutting down to optimal number later
grd_clf = GradientBoostingClassifier(max_depth=2, n_estimators=100,
                                     learning_rate=1.0) # Give any arbit value
print('Training Model..')
grd_clf.fit(X_train, y_train)
print('Done.')
errors = [mean_squared_error(y_test, y_pred) for y_pred in 
          grd_clf.staged_predict(X_test)]
n_estimators_opt = np.argmin(errors) # getting the index of least error
print('Optimal value:', n_estimators_opt)
# create new model with optimal value of n_estimators
grd_clf_opt_1 = GradientBoostingClassifier(max_depth=2,
                                           n_estimators=n_estimators_opt,
                                           learning_rate=1.0)
print('Training Model..')
grd_clf_opt_1.fit(X_train, y_train)
print('Done.')
y_pred = grd_clf_opt_1.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# Implementing actual early-stopping
grd_clf = GradientBoostingClassifier(max_depth=2, warm_start=True) # set warm_start
min_val_error = float('inf')
error_going_up = 0
for n_estimators in range(1, 100):
    grd_clf.n_estimators = n_estimators
    grd_clf.fit(X_train, y_train)
    y_pred = grd_clf.predict(X_test)
    val_error = mean_squared_error(y_test, y_pred)
    print('n_estimators:', n_estimators, 'Error:', val_error)
    if val_error <= min_val_error:
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:
            break # early-stpping
