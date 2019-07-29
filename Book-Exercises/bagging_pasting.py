import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


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
# bagging classifer
bag_clf = BaggingClassifier(DecisionTreeClassifier(),
                            n_estimators=50, bootstrap=True,
                            n_jobs=-1, oob_score=True) # for pasting,
                                                       # make bootstrap=False
print('Training model..')
bag_clf.fit(X_train, y_train)
print('Done')
print('oob score:', bag_clf.oob_score_)
print('Making predictions..')
y_pred = bag_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# sampling features
bag_clf_2 = BaggingClassifier(DecisionTreeClassifier(),
                              n_estimators=50, bootstrap=True,
                              n_jobs=-1, oob_score=True,
                              bootstrap_features=True, max_features=0.5)
print('Training model..')
bag_clf_2.fit(X_train, y_train)
print('Done')
print('oob score:', bag_clf_2.oob_score_)
print('Making predictions..')
y_pred = bag_clf_2.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
RandomForest Classifier
rf_clf = RandomForestClassifier(n_estimators=50, bootstrap=True,
                                max_leaf_nodes=16, n_jobs=-1, oob_score=True)
print('Training model..')
rf_clf.fit(X_train, y_train)
print('Done.')
print('oob score:', rf_clf.oob_score_)
print('Making predictions..')
y_pred = rf_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
# ExtraTree Classifier
ext_clf = ExtraTreesClassifier(n_estimators=50, bootstrap=True,
                               max_leaf_nodes=16, n_jobs=-1, oob_score=True)
print('Training model..')
ext_clf.fit(X_train, y_train)
print('Done.')
print('oob score:', ext_clf.oob_score_)
print('Making predictions..')
y_pred = ext_clf.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
