
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
import pickle
import os
import gc



# pull in data
df = pd.read_csv('train_pre2.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore')
y = df['project_is_approved']
# feature_names = list(X.columns)
#
# pull in test data
df = pd.read_csv('test_pre2.csv')
X_test = df.fillna(0) # std deviation columns for one item prices/counts

df = pd.read_csv('test.csv')
ids = df['id'].values

del df
gc.collect()

# Build the model
cnt = 0
p_buf = []
n_splits = 5
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=0)
auc_buf = []

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))

    model = Pipeline([('scl', StandardScaler()),
                        # ('pca', PCA(n_components=1000)),
                        ('clf', LogisticRegression(C=0.001, solver="sag", max_iter=200, class_weight='balanced'))])

    model.fit(X.loc[train_index], y.loc[train_index])

    # if cnt == 0:
    #     importance = model.feature_importance()
    #     model_fnames = model.feature_name()
    #     tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
    #     tuples = [x for x in tuples if x[1] > 0]
    #     print('Important features:')
    #     for i in range(60):
    #         if i < len(tuples):
    #             print(tuples[i])
    #         else:
    #             break
    #     del importance, model_fnames, tuples

    p = model.predict(X.loc[valid_index])
    auc = roc_auc_score(y.loc[valid_index], p)
    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    pickle.dump(model, open('classifier_pipe_{}.pkl'.format(cnt), 'wb'))
    del model, auc, p
    gc.collect


auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds
subm.to_csv('pipe_submission.csv', index=False)
