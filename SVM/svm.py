# making imports
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import f1_score

# importing dataset
from sklearn import datasets
cancer = datasets.load_breast_cancer()

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.30, random_state=22)
# making svm claddifer
classifier_linear = svm.SVC(kernel='linear')
classifier_gaussian = svm.SVC(kernel='rbf')
# fitting linear kernal
classifier_linear.fit(X_train, y_train)
print('f1_score:', f1_score(y_test, classifier_linear.predict(X_test)))
classifier_gaussian.fit(X_train, y_train)
print('f1_score:', f1_score(y_test, classifier_gaussian.predict(X_test)))
