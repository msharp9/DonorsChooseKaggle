# imports
import re
import numpy as np
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from tqdm import tqdm
import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import gc

# They changed the essay questions on this date
FILTER_DATE = datetime.datetime(2017,5,17,10,0,0,0)

# pull in data
print('Loading data...')
df = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
res = pd.read_csv('resources.csv')

# Filter data due to question change
df = df[df['project_essay_3'].isnull() & df['project_essay_4'].isnull()]

# Convert Dates to Int
df['project_submitted_datetime'] = pd.to_datetime(df['project_submitted_datetime']).values.astype('int64')
test['project_submitted_datetime'] = pd.to_datetime(test['project_submitted_datetime']).values.astype('int64')

# Convert categortical data w/ label encoder
print('Label Encoder...')
cat_cols = ['teacher_prefix', 'school_state', 'project_grade_category', 'project_subject_categories', 'project_subject_subcategories']
for c in tqdm(cat_cols):
    le = LabelEncoder()
    series = np.array(pd.concat([df[c], test[c]], axis=0).astype(str))
    le.fit(series)
    df[c] = le.transform(df[c].astype(str))
    test[c] = le.transform(test[c].astype(str))

del le
gc.collect()

# Pull in price data:
print('Pulling in Resources...')
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
df = df.merge(res, on='id', how='left')
test = test.merge(res, on='id', how='left')

del res
gc.collect()

#Setting up Text Analysis
print('Preparing Text Preprocessing...')
# clean data (remove punctuation and captilization)
def preprocessor(text):
    # text = re.sub("'", '', text)
    text = re.sub('[\W]+', ' ', text.lower())
    return text
# tokenize data (find stem, remove stopwords)
porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]
# stopwords
stop = stopwords.words('english')

text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
n_features = [200,1000,1000,200]
for i,c in tqdm(enumerate(text_columns)):
    tfidf = TfidfVectorizer(max_features=n_features[i], stop_words=stop,
        preprocessor=preprocessor, tokenizer=tokenizer_porter, norm='l2',
    )
    tfidf_train = np.array(tfidf.fit_transform(df[c]).toarray(), dtype=np.float16)
    tfidf_test = np.array(tfidf.transform(test[c]).toarray(), dtype=np.float16)
    vocab = {v: k for k, v in tfidf.vocabulary_.items()}

    for j in range(n_features[i]):
        df[c + '_tfidf_' + vocab[j]] = tfidf_train[:, i]
        test[c + '_tfidf_' + vocab[j]] = tfidf_test[:, i]

    df[c + '_length'] = df[c].apply(lambda x: len(str(x)))
    df[c + '_wc'] = df[c].apply(lambda x: len(str(x).split(' ')))
    test[c + '_length'] = test[c].apply(lambda x: len(str(x)))
    test[c + '_wc'] = test[c].apply(lambda x: len(str(x).split(' ')))

del tfidf, tfidf_train, tfidf_test
gc.collect()

drop_cols = ['id', 'teacher_id', 'project_essay_3', 'project_essay_4', *text_columns]
df.drop(drop_cols, axis=1, errors='ignore', inplace=True)
# y = df['project_is_approved']
ids = test['id'].values
test.drop(drop_cols, axis=1, errors='ignore', inplace=True)
# print(df.head())
# print(test.head())

# Save preprocessing
# df.to_csv('train_pre.csv')
# test.to_csv('test_pre.csv')

# pull in data
df = df.fillna(0) # std deviation columns for one item prices/counts
X = df.drop(['project_is_approved'], axis=1, errors='ignore').values
y = df['project_is_approved'].values
del df
gc.collect()

# put into test and train datasets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.1, stratify=y)

# Normalize
sc = StandardScaler()
X_std = sc.fit_transform(X)
# X_test_std = sc.transform(X_test)

#logistic regression:
# clf = LogisticRegression(C=100.0, random_state=0, penalty='l1', n_jobs=1)
clf = LogisticRegression(solver="sag", max_iter=400)
clf.fit(X_std, y)

# Validation
print('Using cross_val_score')
scores = cross_val_score(estimator=clf,
                         X=X_std,
                         y=y,
                         cv=5,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

pickle.dump(clf, open('classifier_acc_{:.3f}.pkl'.format(np.mean(scores)), 'wb'))

# pull in data
Xt = test.fillna(0).values # std deviation columns for one item prices/counts

# Normalize
Xt_std = sc.transform(Xt)
preds = clf.predict_proba(Xt_std)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds[:,1]
subm.to_csv('submission2.csv', index=False)
