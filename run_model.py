from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from train_model import tf_basic_model

cnn_model = tf_basic_model()

train_dataframe = pd.read_csv('data/train.csv', sep=",", header=None, skiprows=[0])
train_data = train_dataframe[:35000]
eval_data = train_dataframe[35000:42000]

train_label = train_data[0]
train_input_data = train_data.drop(columns=[0])
train_labels = np.asarray(train_label, dtype=np.int32)
predictions = cnn_model.train(train_input_data, train_labels)

eval_label = eval_data[0]
eval_input_data = eval_data.drop(columns=[0])
eval_labels = np.asarray(eval_label, dtype=np.int32)
predictions = cnn_model.evaluate(eval_input_data, eval_labels)

test_dataframe = pd.read_csv('data/test.csv', sep=',', header=None, skiprows=[0])
predictions = cnn_model.predict(test_dataframe)
cnn_model.submit_prediction(predictions)
