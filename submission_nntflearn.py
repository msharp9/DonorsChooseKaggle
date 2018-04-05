import numpy as np
import pandas as pd
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import gc

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


CLASSIFIER = 'classifier_simpleNN.model'

# pull in data
df = pd.read_csv('test_pre2.csv')
df = df.fillna(0) # std deviation columns for one item prices/counts
# print(df.head(), df.shape)

X = df.drop(['project_is_approved'], axis=1, errors='ignore').values
X = X.reshape(-1,X.shape[1],1)

df = pd.read_csv('test.csv')
ids = df['id'].values
del df
gc.collect()

model = neural_network_model(input_size=X.shape[1])
model.load(CLASSIFIER)

preds = model.predict(X, predict_keys="probabilities")

print(preds.shape, preds)

# Prepare submission
subm = pd.DataFrame()
subm['id'] = ids
subm['project_is_approved'] = preds[:,1]
subm.to_csv('submission_nn.csv', index=False)
