# Author: Bunyamin Cetinkaya

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from nn.state_encoder.state_encoder import AbstractStateEncoder

class CNN(AbstractStateEncoder):
    def __init__(self, action_space, state_maxlength, seed=None, config={'cnn_state_dim':64}):
        super(CNN, self).__init__(action_space, state_maxlength, seed=seed, config=config)
        self.cnn_state_dim = config['cnn_state_dim']

    def call(self, input_s, input_f, len_s, name='net'):
        input_state = input_s * input_f
        input_state = tf.expand_dims(input_state, axis=-1 )

        # fewer parameters
        conv_1 = tf.keras.layers.Conv2D(self.cnn_state_dim, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu', kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer())(input_state)
        max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        dense_1 = tf.keras.layers.Flatten(data_format=None)(max_1)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_1)
        # dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)

        # CNN 1
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)

        # CNN 2
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
        # dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(dense)

        # CNN 3
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
        # dense_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense_1)

        # CNN 4
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
        # dense_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense_1)
        # dense_3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(dense_2)

        # CNN 5
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flat)

        # CNN 6
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flat)

        # # CNN 7 best
        # conv_1 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(128, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(512, activation=tf.nn.relu)(flat)

        # CNN 8
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu)(flat)

        # CNN 9
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(2048, activation=tf.nn.relu)(flat)

        # CNN 10
        # conv_1 = tf.keras.layers.Conv2D(32, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(input_state)
        # max_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_1)
        # conv_2 = tf.keras.layers.Conv2D(64, 3, strides=1, padding='valid', data_format=None, dilation_rate=(1, 1), activation='relu')(max_1)
        # max_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)(conv_2)
        # flat = tf.keras.layers.Flatten(data_format=None)(max_2)
        # dense_1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)(flat)
        # dense_2 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense_1)
        # dense_3 = tf.keras.layers.Dense(128, activation=tf.nn.relu)(dense_2)

        output = tf.keras.layers.Dense(self.action_space, activation=self.activation, kernel_initializer=tf.random_normal_initializer(seed=np.random.randint(4096)), bias_initializer = tf.constant_initializer(), name=name)(dense_1)
        
        return output