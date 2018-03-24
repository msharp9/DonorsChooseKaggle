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
res['total_price'] = res['quantity']*res['price']
res = pd.DataFrame(res[['id', 'quantity', 'price', 'total_price']].groupby('id').agg(
        {
            'quantity': ['sum', 'mean', 'std', lambda x: len(np.unique(x)),]
            ,'price': ['count', 'max', 'min', 'mean', 'std', lambda x: len(np.unique(x)),]
            ,'total_price': ['sum', 'max',]
        }
    )).reset_index()
res.columns = ['_'.join(col) for col in res.columns]
res.rename(columns={'id_': 'id'}, inplace=True)
print(res.head())
df = df.merge(res, on='id', how='left')


#Setting up Text Analysis
# clean data (remove punctuation and captilization)
def preprocessor(text):
    # text = re.sub("'", '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    return text

# tokenize data (find stem, remove stopwords)
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

stop = stopwords.words('english')
# def tokenizer_porter(text):
#     stems = [porter.stem(word) for word in text.split() if word not in stop]
#     return stems
#     # return ' '.join(stems)

text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
n_features = [200,1000,1000,200]
print('Excerpt of the dataset', df[text_columns].head(3))
for i,c in tqdm(enumerate(text_columns)):
    tfidf = TfidfVectorizer(max_features=n_features[i], stop_words=stop,
        preprocessor=preprocessor, tokenizer=tokenizer_porter, norm='l2',
    )
    tfidf_train = np.array(tfidf.fit_transform(df[c]).toarray(), dtype=np.float16)

    for j in range(n_features[i]):
        df[c + '_tfidf_' + str(j)] = tfidf_train[:, i]

    df[c + '_length'] = df[c].apply(lambda x: len(str(x)))
    df[c + '_wc'] = df[c].apply(lambda x: len(str(x).split(' ')))

drop_cols = ['id', 'teacher_id', 'project_essay_3', 'project_essay_4', *text_columns]
X = df.drop(drop_cols, axis=1, errors='ignore')
y = df['project_is_approved']
print(x.head(), y.head())


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
