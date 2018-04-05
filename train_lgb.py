
# imports
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, RepeatedKFold
from tqdm import tqdm
import lightgbm as lgb
import pickle
import os
import gc


# pull in data
df = pd.read_csv('train_pre.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
# print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore')
y = df['project_is_approved']
feature_names = list(X.columns)

# pull in test data
df = pd.read_csv('test_pre.csv')
X_test = df.fillna(0) # std deviation columns for one item prices/counts
# print(df.head(), df.shape)

# X_test = df.drop(['Unnamed: 0'], axis=1, errors='ignore')

df = pd.read_csv('test.csv')
ids = df['id'].values

del df
gc.collect()


# Build the model
cnt = 0
p_buf = []
n_splits = 4
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits,
    n_repeats=n_repeats,
    random_state=0)
auc_buf = []

for train_index, valid_index in kf.split(X):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    params = {
        'boosting': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 64,
        'max_depth': 6,
        'learning_rate': 0.01,
        'feature_fraction': 0.5,
        'bagging_fraction': 0.5,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0,
    }

    lgb_train = lgb.Dataset(
        X.loc[train_index],
        y.loc[train_index],
        feature_name=feature_names,
        )
    lgb_train.raw_data = None

    lgb_valid = lgb.Dataset(
        X.loc[valid_index],
        y.loc[valid_index],
        )
    lgb_valid.raw_data = None

    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=10000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        for i in range(60):
            if i < len(tuples):
                print(tuples[i])
            else:
                break
        del importance, model_fnames, tuples

    p = model.predict(X.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y.loc[valid_index], p)
    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(X_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p, dtype=np.float16)
    else:
        p_buf += np.array(p, dtype=np.float16)
    auc_buf.append(auc)

    cnt += 1
    # if cnt > 0: # Comment this to run several folds
    #     break

    del model, lgb_train, lgb_valid, p
    gc.collect


auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds
subm.to_csv('lgb_submission3.csv', index=False)
