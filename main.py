# imports
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# pull in data
df = pd.read_csv('train.csv')
text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary']
print('Excerpt of the dataset', df[text_columns].head(3))

# clean data (remove punctuation and captilization)
def preprocessor(text):
    # text = re.sub("'", '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    return text

# tokenize data (find stem, remove stopwords)
porter = PorterStemmer()
stop = stopwords.words('english')
def tokenizer_porter(text):
    stems = [porter.stem(word) for word in text.split() if word not in stop]
    return ' '.join(stems)

# applying
for column in text_columns:
    df[column].fillna('', inplace=True)
    df[column] = df[column].apply(preprocessor)
    df[column] = df[column].apply(tokenizer_porter)

print(df[text_columns].head(10))

# Save preprocessing
df.to_csv('train_pre.csv')


# Vectorize, Term frequency-inverse document and Normalize
# count = CountVectorizer()
# bag = count.fit_transform(docs)
# print('Vocabulary', count.vocabulary_)
# print('bag.toarray()', bag.toarray())
#
# tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
# print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
#
# # Train on bag of words

# for column in text_columns:
