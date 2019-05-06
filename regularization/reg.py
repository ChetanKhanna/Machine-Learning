# importing modules
import pandas as pd
import math

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

plt.style.use('seaborn')


# defining PolynomialRegression
def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree),
                         LinearRegression(**kwargs))


# defining fuction for plotting learning curves
def learning_curves_(estimator, X, y, train_sizes, cv, scoring):
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=estimator, X=X, y=y, train_sizes=train_sizes,
        cv=cv, scoring=scoring)
    # finding mean of scores
    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)
    plt.plot(train_sizes, train_scores_mean, label='TrainingError')
    plt.plot(train_sizes, validation_scores_mean, label='ValidationError')
    plt.legend()
    plt.title(str(estimator).split('(')[0])

# importing data-set
train_df = pd.read_csv('./train.csv', index_col='ID')
# NOTE: Look at the 'ID' col of train.csv
# and also at the index_col documentation in pandas
# seperate features and labels from data-set
# NOTE: casting to float is neccessary otherwise StandardScaler
# will throw DataConversionWarning
X = train_df.drop('medv', axis=1).astype(float)
y = train_df['medv'].astype(float)
# splitting into test-train
X_train, X_test, y_train, y_test = train_test_split(
                                                    X, y,
                                                    random_state=42,
                                                    test_size=0.30)
# defining model
model = LinearRegression()
model.fit(X_test, y_test)
print('Training accuracy:', model.score(X_train, y_train))
print('Testing accuracy:', model.score(X_test, y_test))
# calculate root-mean-squared-error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print('RMSE:', rmse)

# improving above accuracy using polynomial features
# steps for pipeline -- the last one must be an estimator
steps = [
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', LinearRegression()),
        ]
pipeline = Pipeline(steps)
# training
pipeline.fit(X_train, y_train)
print()
print('Training accuracy:', pipeline.score(X_train, y_train))
print('Testing accuracy:', pipeline.score(X_test, y_test))

# NOTE: If training accuracy is too high compared to Testing accuracy
# it signals an overfit in our training model

# Regularization to avoid pverfitting
# L2 or Ridge regularization
steps = [
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Ridge(alpha=4, fit_intercept=True)),
        ]
# NOTE: alpha value @ 4 seems to be optimal
ridge_pipeline = Pipeline(steps)
ridge_pipeline.fit(X_train, y_train)
print()
print('Training accuracy:', ridge_pipeline.score(X_train, y_train))
print('Testing accuracy:', ridge_pipeline.score(X_test, y_test))

# L1 or Lasso regularization
steps = [
        ('scale', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('model', Lasso(alpha=0.3, fit_intercept=True)),
        ]
# NOTE: alpha va;ue @ 0.3 seems to be optimal
lasso_pipeline = Pipeline(steps)
lasso_pipeline.fit(X_train, y_train)
print()
print('Training accuracy:', lasso_pipeline.score(X_train, y_train))
print('Testing accuracy:', lasso_pipeline.score(X_test, y_test))

# NOTE:  IF The training accuracy < testing accuracy here.
# Possibly because the data-set was too small and the
# regularization was too hard for it. Not sure though, tbh.
# try using smaller value for alpha

# plotting learning curve
train_sizes = [100, 120, 150, 200, 250]
estimators = [Lasso(alpha=0.3, fit_intercept=True),
              LinearRegression(),
              Ridge(alpha=5, fit_intercept=True),
              ]
count = 1
for model in estimators:
    plt.subplot(3, 1, count)
    learning_curves_(model, X, y, train_sizes, 5, 'neg_mean_squared_error')
    count += 1
plt.tight_layout()
plt.show()

# plotting validation curve for deciding alpha for Ridge
alpha_vals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
train_scores, validation_scores = validation_curve(Ridge(), X,
                                                   y, 'alpha', alpha_vals,
                                                   cv=5)
train_scores_mean = train_scores.mean(axis=1)
validation_scores_mean = validation_scores.mean(axis=1)
plt.plot(alpha_vals, train_scores_mean, label='TrainingScore')
plt.plot(alpha_vals, validation_scores_mean, label='ValidationScore')
plt.legend()
plt.show()

# plotting validation curve for deciding alpha for Lasso
alpha_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
train_scores, validation_scores = validation_curve(Lasso(), X,
                                                   y, 'alpha', alpha_vals,
                                                   cv=5)
train_scores_mean = train_scores.mean(axis=1)
validation_scores_mean = validation_scores.mean(axis=1)
plt.plot(alpha_vals, train_scores_mean, label='TrainingScore')
plt.plot(alpha_vals, validation_scores_mean, label='ValidationScore')
plt.legend()
plt.show()

# Deciding degree for Polynomial features
degree_vals = [1, 2, 3, 4, 5, 6, 7]
train_scores, validation_scores = validation_curve(PolynomialRegression(), X,
                                                   y,
                                                   'polynomialfeatures__degree',
                                                   degree_vals, cv=5
                                                   )
train_scores_mean = train_scores.mean(axis=1)
validation_scores_mean = validation_scores.mean(axis=1)
plt.plot(degree_vals, train_scores_mean, label='TrainingScore')
plt.plot(degree_vals, validation_scores_mean, label='ValidationScore')
plt.legend()
plt.show()
