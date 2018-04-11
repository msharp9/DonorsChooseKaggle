
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from helpers.callbacks import TrainingMonitor
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.optimizers import SGD
import pickle
import os
import gc


class simpleNN:
    @staticmethod
    def build(D, x, classes):
        model = Sequential()

        model.add(Dense(D, input_dim=x))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # model.add(Dense(D//2))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        #
        # model.add(Dense(D//4))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))

        # add a softmax layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the model
        return model


# pull in data
df = pd.read_csv('data/train_pre2.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
# print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore').values
y = df['project_is_approved'].values.reshape(-1,1)
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

del df
gc.collect()

# account for skew in the labeled data
classTotals = y.sum(axis=0)
classWeight = classTotals.max() / classTotals
print(classTotals, classWeight)

# put into test and train datasets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y)

# Build and train the model
print("[INFO] training model...")
model = simpleNN.build(32, X.shape[1], 2)
opt = SGD(lr=0.0001)
model.compile(optimizer=opt,
    loss='binary_crossentropy',
    metrics=['accuracy'])
# construct the set of callbacks
path = os.path.sep.join(['output', "{}.png".format(
os.getpid())])
callbacks = [TrainingMonitor(path)]
model.fit(X_train, y_train, epochs=5, batch_size=64
    ,callbacks=callbacks
    ,class_weight=classWeight
    # ,validation_split=0.2
    ,validation_data=(X_test,y_test))

# save the model to file
print("[INFO] serializing model...")
model.save("classifiers/classifier_nn2.model", overwrite=True)
