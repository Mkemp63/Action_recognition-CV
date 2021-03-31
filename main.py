import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import layers, models, callbacks

def make_baseline_model(input_shape):
    model = models.Sequential()

    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(10))

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())
    return model


def train_test_stanford(printing: bool = False):
    with open('./Data/Stanford40/ImageSplits/train.txt', 'r') as f:
        train_files = list(map(str.strip, f.readlines()))
        train_labels = ['_'.join(name.split('_')[:-1]) for name in train_files]
        if printing:
            print(f'Train files ({len(train_files)}):\n\t{train_files}')
            print(f'Train labels ({len(train_labels)}):\n\t{train_labels}\n')

    with open('./Data/Stanford40/ImageSplits/test.txt', 'r') as f:
        test_files = list(map(str.strip, f.readlines()))
        test_labels = ['_'.join(name.split('_')[:-1]) for name in test_files]
        if printing:
            print(f'Test files ({len(test_files)}):\n\t{test_files}')
            print(f'Test labels ({len(test_labels)}):\n\t{test_labels}\n')

    action_categories = sorted(list(set(['_'.join(name.split('_')[:-1]) for name in train_files])))
    if printing:
        print(f'Action categories ({len(action_categories)}):\n{action_categories}')
    return train_files, train_labels, test_files, test_labels


def train_tests_tv(printing: bool = False):
    set_1_indices = [
        [2, 14, 15, 16, 18, 19, 20, 21, 24, 25, 26, 27, 28, 32, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50],
        [1, 6, 7, 8, 9, 10, 11, 12, 13, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 44, 45, 47, 48],
        [2, 3, 4, 11, 12, 15, 16, 17, 18, 20, 21, 27, 29, 30, 31, 32, 33, 34, 35, 36, 42, 44, 46, 49, 50],
        [1, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 22, 23, 24, 26, 29, 31, 35, 36, 38, 39, 40, 41, 42]]
    set_2_indices = [[1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 22, 23, 29, 30, 31, 33, 34, 35, 36, 37, 38, 39],
                     [2, 3, 4, 5, 14, 15, 16, 17, 18, 19, 20, 21, 22, 26, 36, 37, 38, 39, 40, 41, 42, 43, 46, 49, 50],
                     [1, 5, 6, 7, 8, 9, 10, 13, 14, 19, 22, 23, 24, 25, 26, 28, 37, 38, 39, 40, 41, 43, 45, 47, 48],
                     [2, 3, 4, 5, 6, 15, 19, 20, 21, 25, 27, 28, 30, 32, 33, 34, 37, 43, 44, 45, 46, 47, 48, 49, 50]]
    classes = ['handShake', 'highFive', 'hug', 'kiss']  # we ignore the negative class

    # test set
    set_1 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_1_indices[c]]
    set_1_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_1_indices[c]]
    if printing:
        print(f'Set 1 to be used for test ({len(set_1)}):\n\t{set_1}')
        print(f'Set 1 labels ({len(set_1_label)}):\n\t{set_1_label}\n')

    # training set
    set_2 = [f'{classes[c]}_{i:04d}.avi' for c in range(len(classes)) for i in set_2_indices[c]]
    set_2_label = [f'{classes[c]}' for c in range(len(classes)) for i in set_2_indices[c]]
    if printing:
        print(f'Set 2 to be used for train and validation ({len(set_2)}):\n\t{set_2}')
        print(f'Set 2 labels ({len(set_2_label)}):\n\t{set_2_label}')
    return set_1, set_1_label, set_2, set_2_label


def main():
    stanford_x_train, stanford_x_test, stanford_y_train, stanford_y_test = train_test_stanford(True)
    tv_x_train, tv_x_test, tv_y_train, tv_y_test = train_tests_tv(True)
	
	input_shape = (112,112,1)
	model = make_baseline_model(input_shape)


if __name__ == '__main__':
    main()
