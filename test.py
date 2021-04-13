import os

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from tensorflow.keras import layers, models

import HelperFunctions as HF
import OpticalFlow as OptF
import config

def makeTestModel(input_shape):
    model = models.Sequential()

    model.add(layers.Convolution2D(16, (3, 3), input_shape=(32, 32, 3), activation='relu'))  # 30
    model.add(layers.Convolution2D(24, (3, 3), activation='relu'))  # 28
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 14

    model.add(layers.Convolution2D(32, (3, 3), activation='relu'))  # 12
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 6
    model.add(layers.Convolution2D(32, (3, 3), activation='relu'))  # 4
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # 2

    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))

    model.add(layers.Dense(4, activation='softmax'))
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model

# model = makeTestModel(())
def createModel1(input1):
    inputs = layers.Input(shape=input1.shape)

    x = layers.MaxPooling2D(pool_size=(2, 2))(inputs)

    model = models.Model(inputs, x)
    return model


def testje(input1, input2):
    # model1 = models.Sequential()
    # model1.add(layers.MaxPooling2D(pool_size=(2, 2), input_shape=(2, 2, 1)))
    # model1.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
    #               metrics=['accuracy'])
    #
    # model2 = models.Sequential()
    # model2.add(layers.MaxPooling2D(pool_size=(2, 2), input_shape=(2, 2, 1)))
    # model2.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
    #               metrics=['accuracy'])

    inp1 = layers.Input(shape=(2, 2, 1))
    inp2 = layers.Input(shape=(2, 2, 1))

    m1 = layers.MaxPooling2D(pool_size=(2, 2))(inp1)
    m1 = models.Model(inputs=inp1, outputs=m1)

    m2 = layers.MaxPooling2D(pool_size=(2, 2))(inp2)
    m2 = models.Model(inputs=inp2, outputs=m2)

    combined = layers.Concatenate([m1.output, m2.output])

    x = layers.Dense(4, activation='relu')(combined)

    model = models.Model(inputs=[m1.input, m2.input], outputs=x)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())


def test2(input1, input2):
    # inp1 = layers.Input(shape=(2, 2, 1))
    # inp2 = layers.Input(shape=(2, 2, 1))

    m1 = createModel1(input1)
    m2 = createModel1(input2)

    combined = layers.concatenate(axis=3, inputs=[m1.output, m2.output])

    x = layers.Dense(4, activation='relu')(combined)

    model = models.Model(inputs=[m1.input, m2.input], outputs=x)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model



# a = np.array([[1, 2], [3, 4]])
# b = np.array([[5, 6], [7, 8]])
# a = a.reshape((2, 2, 1))
# b = b.reshape((2, 2, 1))
#
# m = createModel1(a)
# m.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
#           metrics=['accuracy'])
# print(m.predict([a]))
# #model = test2(a, b)
# #print(model.predict([a, b]))


# c = np.array([1, 2])
# d = np.array([3, 4])
# e = layers.concatenate([c, d], axis=0)
# print(e)

def make_baseline_model(input_shape, activation1='relu', activation2='relu', activation3='relu',
                        optimizer='adam', hidden_layers=1, hidden_layer_neurons=80, conv_layers=2,
                        filter_size=16, kernel_size=3, dropout=0, output_size=40, k_reg=None):
    model = models.Sequential()
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, input_shape=input_shape,
                            kernel_regularizer=k_reg))
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg)) # 108
    model.add(layers.MaxPooling2D((2, 2)))                                                                      # 54
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1)) # 52
    model.add(layers.MaxPooling2D((2, 2)))  # 26
    for i in range(conv_layers):
        model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1))
    # model.add(layers.Conv2D(32, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg))

    return model

def make_base_model(input_shape, activation1='relu', activation2='relu', activation3='relu',
                        optimizer='adam', hidden_layers=1, hidden_layer_neurons=80, conv_layers=2,
                        filter_size=16, kernel_size=3, dropout=0, output_size=40, k_reg=None):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    x = layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, input_shape=input_shape,
                            kernel_regularizer=k_reg)(x)
    x = layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1)(x)
    x = layers.MaxPooling2D((2, 2))(x)
    for i in range(conv_layers):
        x = layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg)(x)
        x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1)(x)

    model = models.Model(inputs, x)
    return model


def testen(input_shape):
    m1 = make_base_model(input_shape, conv_layers=2)  # m1 = make_baseline_model(input_shape, conv_layers=2)
    m2 = make_base_model(input_shape, conv_layers=2)  # m2 = make_baseline_model(input_shape, conv_layers=2)

    combined = layers.concatenate(axis=3, inputs=[m1.output, m2.output])

    x = layers.Flatten()(combined)
    x = layers.Dense(4, activation='softmax')(x)

    model = models.Model(inputs=[m1.input, m2.input], outputs=x)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                  metrics=['accuracy'])
    print(model.summary())
    return model


# m = make_baseline_model((112, 112, 3), conv_layers=2)
# print(m.summary())
model = testen((112, 112, 3))
input()
# img = cv2.imread("J:\\Python computer vision\\Action_recognition-CV\\Data\\Stanford40\\ImagesConvCrop\\applauding_001.jpg", 1)
# imgf = cv2.imread("J:\\Python computer vision\\Action_recognition-CV\\Data\\Stanford40\\ImagesConvCrop\\applauding_001_flip.jpg", 1)
# train_1 = np.array([img])
# train_2 = np.array([imgf])
# labels = np.array([2])
# print(type(img))
# model.fit(x=[train_1, train_2], y=labels, epochs=10)
#
# print(model.predict(x=[train_1, train_2]))