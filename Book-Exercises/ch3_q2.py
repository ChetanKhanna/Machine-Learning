# import required modules
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
# import sklearn modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


train_filepath = os.path.join(os.path.expanduser('~'), 'Machine Learning',
                              'mnist_train.csv')
test_filepath = os.path.join(os.path.expanduser('~'), 'Machine Learning',
                             'mnist_test.csv')
df_train = pd.read_csv(train_filepath)
df_test = pd.read_csv(test_filepath)
print(df_train.head())
print(df_test.head())
X_train = df_train.iloc[:, 1:].astype('int64')
y_train = df_train.iloc[:, 0].astype('int64')
X_test = df_test.iloc[:, 1:].astype('int64')
y_test = df_test.iloc[:, 0].astype('int64')


# Data augmentation
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dx, dy], cval=0, mode='constant')
    return shifted_image.reshape([-1])


# testing shift_image function
# image = X_train.iloc[1000].values
# shifted_image_down = shift_image(image, 5, 0)
# plt.figure(figsize=(12, 2))
# plt.subplot(121)
# plt.imshow(image.reshape(28, 28), interpolation='nearest', cmap='Greys')
# plt.subplot(122)
# plt.imshow(shifted_image_down.reshape(28, 28), interpolation='nearest',
#            cmap='Greys')
# plt.show()
# Augmenting train set
X_train_vals = X_train.values
y_train_vals = y_train.values
X_test_vals = X_test.values
y_test_vals = y_test.values
print('Data Augmenting..')
for idx in range(X_train.shape[0]):
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        np.append(X_train_vals, shift_image(X_train_vals[idx], dx, dy))
        np.append(y_train_vals, [y_train_vals[idx]])
        print(idx)
print('Data augmentation done.')
# training model
model = KNeighborsClassifier(n_neighbors=4, weights='distance')
print('Training model..')
model.fit(X_train_vals, y_train_vals)
print('Done.')
print(cross_val_score(model, X_test_vals, y_test_vals, cv=3))
