import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras import layers

# importing data
# setting filepaths
filepath_dict = {
    'yelp': './data/sentiment labelled sentences/yelp_labelled.txt',
    'amazon': './data/sentiment labelled sentences/amazon_labelled.txt',
    'imdb': './data/sentiment labelled sentences/imdb_labelled.txt',
}
df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df['source'] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)
# un-comment to preview data files list
# print(df)

# sepertating yelp data-set
df_yelp = df[df['source'] == 'yelp']
sentences = df_yelp['sentence'].values
y = df_yelp['label'].values
# spliiting into test-train data
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)
# vectorize using BOW model
# create a countVectorizer instance for vectorizing
vectorizer = CountVectorizer()
# learning features by creating a vocabulary of all words in all sentences
vectorizer.fit(sentences_train)
# transforming learnt features into an array format
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
# Applying logistic regression
classifier = LogisticRegression()
# fitting training data
classifier.fit(X_train, y_train)
# testing classifier accuracy on test data
accuracy = classifier.score(X_test, y_test)
print(accuracy)

# seperating amazon data
df_amazon = df[df['source'] == 'amazon']
sentences = df_amazon['sentence'].values
y = df_amazon['label'].values
# splitting into test and train data set
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.30, random_state=1000)
# vectorize sentences using BOW model
vectorizer = CountVectorizer()
# learning features
vectorizer.fit(sentences_train)
# transforming to array
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
# applying logistic regression
classifier = LogisticRegression()
# fitting data
classifier.fit(X_train, y_train)
# finding accuracy
accuracy = classifier.score(X_test, y_test)
print(accuracy)

# seperating imdb data
df_imdb = df[df['source'] == 'imdb']
sentences = df_imdb['sentence'].values
y = df_imdb['label'].values
# splitting data
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.30, random_state=1000)
# vectorize sentence to turn into features
vectorizer = CountVectorizer()
vectorizer.fit(sentences_train)
X_train = vectorizer.transform(sentences_train)
X_test = vectorizer.transform(sentences_test)
# making classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy)

# using Keras to improve accuracy
input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# configuring keras
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy'])
model.summary()
# Training with Keras
history = model.fit(
    X_train, y_train,
    epochs=50,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10)
loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print(accuracy)
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(accuracy)
