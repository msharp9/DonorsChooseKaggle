import numpy as np
import pandas as pd
import keras.models as models
import gc

from keras import backend as K
import tensorflow as tf

# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
    return TP/P



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
# model = models.load_model("classifiers/classifier_nn2.model", custom_objects={'auc': auc})
preds = model.predict(X, batch_size=64)

print(preds.shape, preds)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds
subm.to_csv('submissions/submission_nn2.csv', index=False)
