# imports
import re
import numpy as np
import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import datetime

# They changed the essay questions on this date
FILTER_DATE = datetime.datetime(2017,5,17,10,0,0,0)

# pull in data
df = pd.read_csv('train.csv')
res = pd.read_csv('resources.csv')

# Filter data due to question change
print(df.shape)
df = df[df['project_essay_3'].isnull() & df['project_essay_4'].isnull()]
print(df.shape)

# Convert Dates to Int
df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime']).values.astype(np.int64)
print('Dates:', df['project_submitted_datetime'].head(3))

# Convert categortical data w/ label encoder
print('Label Encoder...')
cat_cols = ['teacher_id', 'teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']
for c in tqdm(cat_cols):
    le = LabelEncoder()
    df[c] = le.fit_transform(df[c].astype(str))
    # test[c] = le.transform(test[c].astype(str))
print('Label Encoded Cols:', df[cat_cols].head(3))

# Pull in price data:
res = pd.DataFrame(res[['id', 'price']].groupby('id').price.agg(
        ['sum', 'count', 'max', 'mean', 'std', lambda x: len(np.unique(x)),]
    )).reset_index()
print(res.head())
df = df.merge(res, on='id', how='left')



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

# # applying
# for column in tqdm(text_columns):
#     df[column].fillna('', inplace=True)
#     df[column] = df[column].apply(preprocessor)
#     df[column] = df[column].apply(tokenizer_porter)
#
# print(df[text_columns].head(10))
#
# # Save preprocessing
# df.to_csv('train_pre.csv')


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
