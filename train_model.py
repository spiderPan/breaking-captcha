from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)


class tf_basic_model:
    def __init__(self):
        mnist_classifier = tf.estimator.Estimator(model_fn=self.cnn_model_fn,
                                                  model_dir="/tmp/mnist_convnet_model_v2")
        self.model = mnist_classifier

    def cnn_model_fn(self, features, labels, mode):
        input_layer = tf.reshape(features['x'], [-1, 28, 28, 1])

        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

        conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        drop1 = tf.layers.dropout(inputs=pool1, rate=0.25)

        conv3 = tf.layers.conv2d(
            inputs=drop1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu
        )
        pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)
        drop2 = tf.layers.dropout(inputs=pool2, rate=0.25)

        pool2_flat = tf.reshape(drop2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

        predictions = {
            'classes': tf.argmax(input=logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name="softmax_tensor")
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
        }

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

    def train(self, train_input_data, train_labels):
        tensors_to_log = {'probabilities': 'softmax_tensor'}
        logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.asarray(train_input_data, dtype=np.float32)},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True
        )

        self.model.train(
            input_fn=train_input_fn,
            steps=20000,
            hooks=[logging_hook]
        )

    def evaluate(self, eval_input_data, eval_labels):
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.asarray(eval_input_data, dtype=np.float32)},
            y=eval_labels,
            num_epochs=1,
            shuffle=False
        )
        eval_results = self.model.evaluate(input_fn=eval_input_fn)
        print(eval_results)

    def predict(self, predict_input_data):
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': np.asarray(predict_input_data, dtype=np.float32)},
            num_epochs=1,
            shuffle=False
        )
        predict_results = self.model.predict(input_fn=test_input_fn)
        return predict_results

    def submit_prediction(self, predictions, filename=None):
        if filename is None:
            filename = 'submission'
        predictions = np.array([item['classes'] for item in predictions])
        submission = pd.DataFrame()
        submission['Label'] = predictions
        submission['ImageId'] = submission.index + 1
        submission = submission.reindex(columns=['ImageId', 'Label'])
        submission.head()
        submission.to_csv('./data/' + filename + '.csv', index=False)
