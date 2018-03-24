import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())


print('TFIDF...')
tfidf = TfidfVectorizer(max_features=3)
tfidf.fit(docs)
X = tfidf.transform(docs)
print(tfidf.vocabulary_, tfidf.idf_)
print(X.shape)
print(X.toarray())
vocab = {v: k for k, v in tfidf.vocabulary_.items()}
for i in range(3):
    print(i)
    print(vocab[i])

import datetime
print(datetime.datetime(2017,5,17,10,0,0,0))


import time
from datetime import datetime
timestamp = int(time.mktime(datetime.now().timetuple()))
now = datetime.fromtimestamp(timestamp)
print(timestamp,now)

text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_essay_3', 'project_essay_4', 'project_resource_summary']
drop_cols = ['id', 'teacher_id', *text_columns]
print(drop_cols)

df = pd.read_csv('train.csv')
X = df.drop(drop_cols, axis=1, errors='ignore')
print(list(df), list(X))
