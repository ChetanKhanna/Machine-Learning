import  pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
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
