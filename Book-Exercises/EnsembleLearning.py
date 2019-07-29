# importing required packages
import pandas as pd
# importing sklearn packages
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


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
# classifiers
lr_clf = LogisticRegression()
svm_clf = SVC()
rf_clf = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[('lr', lr_clf),
                              ('svm', svm_clf), ('rf', rf_clf)],
                              voting='hard')
print('Running models and predictions..')
# printing individual score
for clf in (lr_clf, svm_clf, rf_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Moons dataset
# making dataset
X, y = make_moons(n_samples=20000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)
# classifiers
lr_clf = LogisticRegression()
svm_clf = SVC()
rf_clf = RandomForestClassifier()
voting_clf = VotingClassifier(estimators=[('lr', lr_clf),
                              ('svm', svm_clf), ('rf', rf_clf)],
                              voting='hard')
print('Running models and predictions..')
# printing individual score
for clf in (lr_clf, svm_clf, rf_clf, voting_clf):
    print('Training', clf.__class__.__name__, '...')
    clf.fit(X_train, y_train)
    print('Predicting labels from model')
    y_pred = clf.predict(X_test)
    print('Getting accuracy')
    print(accuracy_score(y_test, y_pred))
