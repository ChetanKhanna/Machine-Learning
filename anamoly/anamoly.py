import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# loading dataset from .mat files
# NOTE: These X are features which take on very large or very small
# values for an anamolous example. Xval, yval are for validation set.
mat = loadmat('./ex8data1.mat')
X = mat['X']
Xval = mat['Xval']
yval = mat['yval']
# Visualizing dataset
plt.scatter(X[:, 0], X[:, 1], marker='x', s=10)
plt.xlabel('Latency')
plt.ylabel('Throuput')
plt.show()


def eastimate_gaussian_params(X):
    '''
    Estimates the gaussian distribution for each feature
    in matrix X -- fitting paramers mu and sigme2 on all features in X
    '''
    m = X.shape[0]
    # estimating parameters
    # estimate mean
    sum_ = np.sum(X, axis=0)
    mu = 1/m*sum_
    # estimate variance
    var = 1/m*np.sum((X - mu)**2, axis=0)
    return mu, var


mu, sigma2 = eastimate_gaussian_params(X)


def find_probability(X, mu, sigma2):
    '''
    computes the probability density of X under
    multivariate gaussian.
    '''
    k = len(mu)
    sigma2 = np.diag(sigma2)
    X = X - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma2)**0.5)) * \
        np.exp(-0.5 * np.sum(X @ np.linalg.pinv(sigma2) * X, axis=1))
    return p


p = find_probability(X, mu, sigma2)
# plotting contours
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker='x')
x_axis, y_axis = np.meshgrid(np.linspace(0, 35, num=70),
                             np.linspace(0, 35, num=70))
p2 = find_probability(np.hstack((x_axis.flatten()[:, np.newaxis],
                      y_axis.flatten()[:, np.newaxis])), mu, sigma2)
contour_level = 10**np.array([np.arange(-2, 0, 3, dtype=np.float)]).T
plt.contour(x_axis, y_axis, p2[:, np.newaxis].reshape(x_axis.shape),
            [-0.5, 0.0048, 0.267, 0.329, 1, 1.5, 2.6, 5])
plt.xlim(0, 30)
plt.ylim(0, 30)
plt.show()


def select_threshold(y, p):
    '''
    Select epsilon for outlier
    This also acts as an evaluation metric for anamoly detection
    algorithm. Check it for different X's and judge on F1 score returned
    as to which combination of features in X is best suited.
    '''
    best_epi = 0
    best_F1 = 0
    stepsize = (max(p) - min(p))/1000
    epi_range = np.arange(p.min(), p.max(), stepsize)
    for epi in epi_range:
        predictions = (p < epi)[:, np.newaxis]
        true_pos = np.sum(predictions[y == 1] == 1)
        false_pos = np.sum(predictions[y == 0] == 1)
        false_neg = np.sum(predictions[y == 1] == 0)
        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        F1 = (2*precision*recall)/(precision + recall)
        if F1 > best_F1:
            best_F1 = F1
            best_epi = epi
    return best_epi, best_F1


p = find_probability(Xval, mu, sigma2)
epsilon, F1 = select_threshold(yval, p)
print("Best epsilon found using cross-validation:", epsilon)
print("Best F1 on Cross Validation Set:", F1)

# Repeating for high dimensional dataset
mat2 = loadmat('./ex8data2.mat')
X2 = mat2['X']
Xval2 = mat2['Xval']
yval2 = mat2['yval']
# finding mean and sigma
mu2, sigma2_2 = eastimate_gaussian_params(X2)
# traingin-set
p_2 = find_probability(X2, mu2, sigma2_2)
# CV-set
pval2 = find_probability(Xval2, mu2, sigma2_2)
# select the best threshold
epsilon2, F1_2 = select_threshold(yval2, pval2)
# results
print('Best F1 score:', F1_2)
print('Best epsilon on CV set:', epsilon2)
print('Number of outliers:', np.sum(p_2 < epsilon2))
