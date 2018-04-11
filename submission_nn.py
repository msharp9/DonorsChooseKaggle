import numpy as np
import pandas as pd
import keras.models as models
import gc


# pull in data
df = pd.read_csv('data/test_pre2.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
# print(df.head(), df.shape)

X = df.values
df = pd.read_csv('data/test.csv')
ids = df['id'].values
del df
gc.collect()

model = models.load_model("classifiers/classifier_nn2.model")
preds = model.predict(X)

print(preds.shape, preds)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds[:,1]
subm.to_csv('submissions/submission_nn2.csv', index=False)
