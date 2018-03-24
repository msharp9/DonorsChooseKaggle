import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(bag).toarray())


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
