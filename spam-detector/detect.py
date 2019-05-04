import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


df = pd.read_csv('./spam.csv', encoding='latin-1')
X = df['v2'].values
y = df['v1'].values
# train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.40,
                                                    random_state=42)
sentences_train, sentences_cv, y_train, y_cv = train_test_split(
                                                                sentences_test,
                                                                y_test,
                                                                test_size=0.50,
                                                                random_state=22
                                                                )
# model training
# vectorizing to get features from text
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
X_cv = vectorizer.transform(sentences_cv)
# performing classification
classifier = RidgeClassifier(alpha=0)
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
print(classifier.predict(X_test))
# estimating optimal alpha
alpha = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
model_ = RidgeClassifierCV(alphas=alpha)
model_.fit(X_cv, y_cv)
print(model_.score(X_test, y_test))
print(model_.alpha_)
