# This is a spam classifier I build while going through the
# exercisied in the book Hands on ML with Sklearn and Tensorflow.

# Importing required modules
import pandas as pd
import matplotlib.pyplot as plt
# Importing Sklearn functionalities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_curve


filepath = './spam.csv'
df = pd.read_csv(filepath, encoding='latin-1')
print(df.head())
X = df['v2'].values
y = df['v1'].values
# Splitting into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# One-hot-encoding features
vectorizer = CountVectorizer(binary=True)
vectorizer.fit(X_train)  # creates a feature dict
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print(X_train_vec.shape, len(X_train))
# Encoding the labels
encoder = LabelEncoder()
print(y_test)
y_train_enc = encoder.fit_transform(y_train)
y_test_enc = encoder.fit_transform(y_test)
print((y_train_enc[0:10]))
print(y_train[0:10])

# Training and testing SGDClassifier
classifier_1 = SGDClassifier()
classifier_1.fit(X_train_vec, y_train_enc)
print(cross_val_score(classifier_1, X_test_vec, y_test, cv=3))
y_pred = cross_val_predict(classifier_1, X_train_vec, y_train_enc)
print('f1_score for classifier_1: ', f1_score(
    y_train_enc, y_pred))
print('precision for classifier_1: ', precision_score(
    y_train_enc, y_pred))
print('recall for classifier_1: ', recall_score(
    y_train_enc, y_pred))
precisions, recalls, threshholds = precision_recall_curve(
    y_train_enc, y_pred)
plt.plot(threshholds, precisions[:-1], 'b--', label='precision')
plt.plot(threshholds, recalls[:-1], 'g-', label='recall')
plt.ylim([0, 1])
plt.legend()
plt.show()

# Training and testing RidgeClassifier
classifier_2 = RidgeClassifier()
classifier_2.fit(X_train_vec, y_train_enc)
print(cross_val_score(classifier_2, X_test_vec, y_test_enc, cv=3))
y_pred = cross_val_predict(classifier_2, X_train_vec, y_train_enc)
print('f1_score for classifier_2: ', f1_score(
    y_train_enc, y_pred))
print('precision for classifier_2: ', precision_score(
    y_train_enc, y_pred))
print('recall for classifier_2: ', recall_score(
    y_train_enc, y_pred))
precisions, recalls, threshholds = precision_recall_curve(
    y_train_enc, y_pred)
plt.plot(threshholds, precisions[:-1], 'b--', label='precision')
plt.plot(threshholds, recalls[:-1], 'g-', label='recall')
plt.ylim([0, 1])
plt.legend()
plt.show()

# Traning and Testing RandomForsetClassifier
classifier_3 = RandomForestClassifier()
classifier_3.fit(X_train_vec, y_train_enc)
print(cross_val_score(classifier_3, X_test_vec, y_test_enc, cv=3))
y_pred = cross_val_predict(classifier_3, X_train_vec, y_train_enc)
print('f1_score for classifier_3: ', f1_score(
    y_train_enc, y_pred))
print('precision for classifier_3: ', precision_score(
    y_train_enc, y_pred))
print('recall for classifier_3: ', recall_score(
    y_train_enc, y_pred))
precisions, recalls, threshholds = precision_recall_curve(
    y_train_enc, y_pred)
plt.plot(threshholds, precisions[:-1], 'b--', label='precision')
plt.plot(threshholds, recalls[:-1], 'g-', label='recall')
plt.ylim([0, 1])
plt.legend()
plt.show()

# Trainind and testing KNeighboursClassifiers
classifier_4 = KNeighborsClassifier()
classifier_4.fit(X_train_vec, y_train_enc)
print(cross_val_score(classifier_4, X_test_vec, y_test_enc, cv=3))
y_pred = cross_val_predict(classifier_4, X_train_vec, y_train_enc)
print('f1_score for classifier_4: ', f1_score(
    y_train_enc, y_pred))
print('precision for classifier_4: ', precision_score(
    y_train_enc, y_pred))
print('recall for classifier_4: ', recall_score(
    y_train_enc, y_pred))
precisions, recalls, threshholds = precision_recall_curve(
    y_train_enc, y_pred)
plt.plot(threshholds, precisions[:-1], 'b--', label='precision')
plt.plot(threshholds, recalls[:-1], 'g-', label='recall')
plt.ylim([0, 1])
plt.legend()
plt.show()
