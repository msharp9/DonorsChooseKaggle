
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle
import os


# Just a simple multilayer perceptron model from tflearn
# input size would be 4 since there are four observations every action
def neural_network_model(input_size):
    network = input_data(shape=[None, input_size, 1], name='input')

    network = fully_connected(network, 1024, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 256, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 128, activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 64, activation='relu')
    network = dropout(network, 0.8)

    # 2 is for left or right - since we are trying to predict which action should be taken from the observations
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', learning_rate=1e-4, loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(network)

    return model



# pull in data
df = pd.read_csv('train_pre2.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore').values
y = df['project_is_approved'].values.reshape(-1,1)

X = X.reshape(-1,X.shape[1],1)
ohe = OneHotEncoder()
ohe.fit(y)
y = ohe.transform(y).toarray()
# y = np.array(y)
# put into test and train datasets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, stratify=y)

# print(X, X.shape)
# print(y, y.shape)
model = neural_network_model(input_size=X.shape[1])
model.fit(X, y, n_epoch=3, snapshot_step=5000, show_metric=True)

model.save('classifier_simpleNN.model')


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
