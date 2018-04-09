import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
#
# count = CountVectorizer()
# docs = np.array([
#         'The sun is shining',
#         'The weather is sweet',
#         'The sun is shining, the weather is sweet, and one and one is two'])
# bag = count.fit_transform(docs)
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
# print(tfidf.fit_transform(bag).toarray())
#
# def preprocessor(text):
#     # text = re.sub("'", '', text)
#     text = re.sub('[\W]+', ' ', text.lower())
#     return text
# # tokenize data (find stem, remove stopwords)
# porter = PorterStemmer()
# def tokenizer_porter(text):
#     return [porter.stem(word) for word in text.split()]
# stop = stopwords.words('english')
#
# print('TFIDF...')
# tfidf = TfidfVectorizer(max_features=3,
#     stop_words=stop,
#     preprocessor=preprocessor,
#     tokenizer=tokenizer_porter,
#     norm='l2',)
# tfidf.fit(docs)
# X = tfidf.transform(docs)
# print(tfidf.vocabulary_, tfidf.idf_)
# print(X.shape)
# print(X.toarray())
# vocab = {v: k for k, v in tfidf.vocabulary_.items()}
#
# df = pd.DataFrame(docs)
# print(df)
# X = np.array(X.toarray(), dtype=np.float16)
# for i in range(3):
#     print(i)
#     print(vocab[i])
#     print(X[:, i])
#     df['tfidf_' + vocab[i]] = X[:, i]
# print(df)
#
#
# import datetime
# print(datetime.datetime(2017,5,17,10,0,0,0))
#
#
# import time
# from datetime import datetime
# timestamp = int(time.mktime(datetime.now().timetuple()))
# now = datetime.fromtimestamp(timestamp)
# print(timestamp,now)
#
# text_columns = ['project_title', 'project_essay_1', 'project_essay_2', 'project_resource_summary']
# drop_cols = ['id', 'teacher_id', *text_columns]
# print(drop_cols)
#
# df = pd.read_csv('train.csv')
# X = df.drop(drop_cols, axis=1, errors='ignore')
# print(list(df), list(X))
#
#
# docs = df[text_columns][:5]
# print(docs)
#
# print('TFIDF...')
# df2 = pd.DataFrame(docs)
# for c in text_columns:
#     tfidf = TfidfVectorizer(max_features=10,
#         stop_words=stop,
#         preprocessor=preprocessor,
#         tokenizer=tokenizer_porter,
#         norm='l2',)
#     docs = df[c][:5]
#     tfidf.fit(docs)
#     X = tfidf.transform(docs)
#     vocab = {v: k for k, v in tfidf.vocabulary_.items()}
#     X = np.array(X.toarray(), dtype=np.float16)
#     for i in range(10):
#         df2[c+'_tfidf_' + vocab[i]] = X[:, i]
#     print(df2)

scores = [10,20.5,30.1]
print('classifier_acc_{:.3f}_{:.3f}.pkl'.format(np.mean(scores),np.std(scores)))




from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Dense
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

print(data,labels,data.shape,labels.shape)
# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)
