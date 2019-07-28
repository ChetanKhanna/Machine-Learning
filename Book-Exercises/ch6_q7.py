# importing sklearn packages
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)
model = DecisionTreeClassifier(max_leaf_nodes=5)
model.fit(X_train, y_train)
print('Accuracy: ', model.score(X_test, y_test))
parameters = {'max_depth':[2, 3, 4, 5], 'max_leaf_nodes':[2, 3, 4, 5]}
search = GridSearchCV(model, parameters, cv=3, n_jobs=-1)
search.fit(X_train, y_train)
print('best_score: ', search.best_score_)
print('best_params: ', search.best_params_)
