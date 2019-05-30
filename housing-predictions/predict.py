import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('./housing.csv')
# getting a look at the dataset
# head() - gives a peek at first 5 rows of df
# infor() - gives summary of each feature(column) - total
#           number of instances and type of data
# print(df.head())
# print(df.info())
# counting number of non-numeric data column
# for each categorical feature, values_counts() gives
# number of occurances of each catefory
# print(df['ocean_proximity'].value_counts())
# numerical attributes
# describe() gives some mathematical insight into
# All the numerical features in the dataset
# print(df.describe())
# plotting histogram for features
df.hist()
plt.tight_layout()
plt.show()

# splitting data into train-test -- normal split
train_set, test_set = train_test_split(df, test_size=0.25, random_state=42)

# stratified split
# adding income category feature for stratified splitting
df['income_category'] = np.ceil(df['median_income'] / 1.5)
df['income_category'].where(df['income_category'] < 5, 5.0, inplace=True)
# stratified splitting
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, test_index in split.split(df, df['income_category']):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]
# removing 'income_category' from df
for set_ in (strat_train_set, strat_test_set):
    set_.drop(['income_category'], axis=1, inplace=True)

# plotting a scatter graph longitude vs latitude
# the more dense places are more populated
df.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
plt.show()

# getting correlation matrix for df
corr_mat = df.corr()
# print(corr_mat)
# Testing new attributes and its correlation
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
# print(df.info())
corr_mat = df.corr()
# print(corr_mat['median_house_value'])

# getting features and labels
X_train = strat_train_set.drop('median_house_value', axis=1)
y_train = strat_train_set['median_house_value'].copy()
X_test = strat_test_set.drop('median_house_value', axis=1)
y_test = strat_test_set['median_house_value'].copy()
# Data cleaning
# We have two taska at hand:
# 1. do something about the missing values
# 2. Scale the numerical values
# 3. Convert non-numerical(categorical) values to numeric

# for 1. we have 3 options -- dropna(), drop(), fillna()
# We use fillna() and we fill missing values with median of
# training set. We can do this manually as well as use imputer()
# imputer = SimpleImputer(strategy='median')
# # Dropping categorical values since imputer with
# # median can only works on numeric data
X_train_num = X_train.drop("ocean_proximity", axis=1)
# imputer.fit(X_train_num)
# X_train_num = imputer.transform(X_train_num)

# # Handling categorical data
# X_train_cat = X_train['ocean_proximity'].copy()
# encoder = OneHotEncoder()
# X_train_cat = encoder.fit_transform(X_train_cat)

# # Feature Scaling
# scaler = StandardScaler()
# scaler.fit(X_train_num)
# X_train_num = scaler.transform(X_train_num)

# # printing results
# print(X_train_num)
# print(X_train_cat)


# Combining features
# We need two seperate pipelines to handle text and
# numerical data. We cannot directly pass all transformers
# into FeatureUnion Pipeline
# Writting a new transformer to segregate pandas DF
class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes):
        self.attributes = attributes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes].values


num_attr = list(X_train_num)
cat_attr = ['ocean_proximity']
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attr)),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attr)),
    ('encoder', OneHotEncoder()),
    ])
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
    ])
X_train_transformed = full_pipeline.fit_transform(X_train)
# print(X_train_transformed)
# print(X_train_transformed.shape)
X_test_transformed = full_pipeline.transform(X_test)

# Training a model
model_1 = LinearRegression()
model_1.fit(X_train_transformed, y_train)
print('accuracy:', model_1.score(X_test_transformed, y_test))
print()

model_2 = DecisionTreeRegressor()
model_2.fit(X_train_transformed, y_train)
print('accuracy:', model_2.score(X_test_transformed, y_test))
print()

model_3 = RandomForestRegressor(n_estimators=20)
model_3.fit(X_train_transformed, y_train)
print('accuracy:', model_3.score(X_test_transformed, y_test))
