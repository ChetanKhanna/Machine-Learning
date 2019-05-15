import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# classification using DecisionTree
DB = pd.read_csv('./bill_authentication.csv')
X = DB.drop('Class', axis=1)
y = DB['Class']
# splitting DB
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=22)
# model training and predicting
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# evaluating algorithm
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Regression using DecisionTree
DB = pd.read_csv('./petrol_consumption.csv')
X = DB.drop('Petrol_Consumption', axis=1)
y = DB['Petrol_Consumption']
# splitting DB
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0)
# model training and testing
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
# evaluating
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)
