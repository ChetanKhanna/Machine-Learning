import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

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
