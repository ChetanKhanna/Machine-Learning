import pandas as pd

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
# un-comment to preview data files list
df = pd.concat(df_list)
print(df)
