"""Neual network classifier model in keras."""
import warnings
import numpy as np
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential
warnings.filterwarnings("ignore")


rn.seed(12345)
np.random.seed(42)

session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
tf.compat.v1.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)


def get_model(input_dim, hidden_dims, output_dim):
    """Get a model with given params."""
    model = Sequential()
    model.add(Dense(hidden_dims[0], input_dim=input_dim, activation='relu'))
    for i in range(1, len(hidden_dims)):
        model.add(Dense(hidden_dims[i], activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model
