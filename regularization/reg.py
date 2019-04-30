# importing modules
import pandas as pd
import numpy as np
import math

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


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
