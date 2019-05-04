import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv('./spam.csv', encoding='latin-1')
X = df['v2'].values
y = df['v1'].values
# train-test split
sentences_train, sentences_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.30,
                                                    random_state=42)
# model training
# vectorizing to get features from text
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
# performing classification
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
print(classifier.predict(X_test))
