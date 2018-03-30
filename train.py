
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
import os



# pull in data
df = pd.read_csv('train_pre.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
print(df.head(), df.shape)

X = df.drop(['Unnamed: 0','project_is_approved'], axis=1, errors='ignore').values
y = df['project_is_approved'].values

# put into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y)

# Normalize
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)


#logistic regression:
# clf = LogisticRegression(C=100.0, random_state=0, penalty='l1', n_jobs=1)
clf = LogisticRegression(C=0.01, solver="sag", max_iter=400)
clf.fit(X_train_std, y_train)

# clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
# doc_stream = stream_docs(path='./movie_data.csv')
#
# pbar = pyprind.ProgBar(45)
#
# classes = np.array([0, 1])
# for _ in range(45):
#     X_train, y_train = get_minibatch(doc_stream, size=1000)
#     if not X_train:
#         break
#     X_train = vect.transform(X_train)
#     clf.partial_fit(X_train, y_train, classes=classes)
#     pbar.update()
#
#
# X_test, y_test = get_minibatch(doc_stream, size=5000)
# X_test = vect.transform(X_test)
# print('Accuracy: %.3f' % clf.score(X_test, y_test))
#
# clf = clf.partial_fit(X_test, y_test)


# Validation
print('Using cross_val_score')
scores = cross_val_score(estimator=clf,
                         X=X_train_std,
                         y=y_train,
                         cv=3,
                         n_jobs=1)
print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

# # Confusion Matrix
# y_pred = clf.predict(X_test_std)
# confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# print('Confusion matrix', confmat)
#
# fig, ax = plt.subplots(figsize=(2.5, 2.5))
# ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
# for i in range(confmat.shape[0]):
#     for j in range(confmat.shape[1]):
#         ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
# plt.xlabel('predicted label')
# plt.ylabel('true label')
# # plt.tight_layout()
# # plt.savefig('./figures/confusion_matrix.png', dpi=300)
# plt.show()

pickle.dump(clf, open('classifier_acc_{:.3f}.pkl'.format(np.mean(scores)), 'wb'))
