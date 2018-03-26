import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

CLASSIFIER = 'classifier_acc_.846.pkl'

# pull in data
df = pd.read_csv('test_pre.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
print(df.head(), df.shape)

X = df.drop(['Unnamed: 0'], axis=1, errors='ignore').values

# Normalize
sc = StandardScaler()
X_std = sc.fit_transform(X)

clf = pickle.load(open(CLASSIFIER, 'rb'))
preds = clf.predict_proba(X_std)

print(preds.shape, preds)

# Prepare submission
subm = pd.DataFrame()
# subm['id'] = id_test --oops I dropped these
subm['project_is_approved'] = preds[:,1]
subm.to_csv('submission.csv', index=False)
