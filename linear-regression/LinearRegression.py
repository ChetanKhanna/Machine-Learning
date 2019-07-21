from os.path import expanduser, join
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Loading training data
HOME = expanduser('~')
path = 'Machine Learning/linear-regression'
fname = join(HOME, path, './single-feature.txt')
X, y = np.loadtxt(fname, delimiter=',', unpack=True)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
# creating model
model = LinearRegression()
# Fitting training data
model.fit(X, y)
# Printing theta0 and theta1
# intercept_ --> theta0 and coef_ --> list of [theta1, theta2, ...]
print(model.intercept_, model.coef_)
# Coeff of determination for training data
print(model.score(X, y))
# predicting for x = 3.5 and x = 7.0
print(model.predict(np.array([3.5, 7.0]).reshape(-1, 1)))

# Multi-feature training set
fname = join(HOME, path, './multi-feature.txt')
X1, X2, y = np.loadtxt(fname, delimiter=',', unpack=True)
# X = np.hstack((X1, X2))
X = np.c_[X1, X2]
X = X.reshape(-1, 2)
y = y.reshape(-1, 1)
model.normalize = True
model.fit(X, y)
print(model.intercept_, model.coef_, model.score(X, y))
print(model.predict(np.array([1650, 3]).reshape(-1, 2)))

# Using Polynomial features
X_ = PolynomialFeatures(degree=2, include_bias=True).fit_transform(X)
model.fit(X_, y)
print(model.intercept_, model.coef_, model.score(X_, y))
