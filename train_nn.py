
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from helpers.callbacks import TrainingMonitor
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.core import Dense
import pickle
import os
import gc


class simpleNN:
    @staticmethod
    def build(D, x, classes):
        model = Sequential()

        model.add(Dense(D, input_dim=x))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(D))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(D))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        # add a softmax layer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the model
        return model


# pull in data
df = pd.read_csv('data/train_pre.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore').values
y = df['project_is_approved'].values.reshape(-1,1)
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()

del df
gc.collect()

# # put into test and train datasets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y)

# Build and train the model
print("[INFO] training model...")
model = simpleNN.build(1032, X.shape[1], 2)
model.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# construct the set of callbacks
path = os.path.sep.join(['output', "{}.png".format(
os.getpid())])
callbacks = [TrainingMonitor(path)]
model.fit(X, y, epochs=10, batch_size=32, callbacks=callbacks, validation_split=0.2)

# save the model to file
print("[INFO] serializing model...")
model.save("classifiers/classifier_nn.model", overwrite=True)
