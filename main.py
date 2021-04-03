import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from tensorflow.keras import layers, models

import config


def transfer_learn_model(model_path, output_layer):
    old_model = models.load_model(model_path)
    model = models.Sequential()

    for layer in old_model.layers[:-1]:
        model.add(layer)

    for layer in model.layers:
        layer.trainable = False

    model.add(output_layer)
    model.compile()
    print(model.summary())
    return model


def make_baseline_model(input_shape, activation1='relu', activation2='relu', activation3='relu',
                        optimizer='adam', hidden_layers=1, hidden_layer_neurons=80, conv_layers=2,
                        filter_size=16, kernel_size=3, dropout=0):
    model = models.Sequential()
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    for i in range(conv_layers):
        model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1))
    model.add(layers.Flatten())
    for i in range(hidden_layers):
        model.add(layers.Dense(hidden_layer_neurons, activation=activation2))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(40, activation=activation3))
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # print(model.summary())
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


def preprocess(stanford_x_imgs, save: bool = False, aug: bool = True):
    lijst = []
    for fileName in stanford_x_imgs:
        img = cv2.imread(config.STANF_IMG + fileName, 0)
        img = cv2.resize(img, (config.Image_size, config.Image_size))
        if aug and save:
            img_flip = cv2.flip(img, 1)
            cv2.imwrite(config.STANF_CONV + fileName[:-4] + "_flip.jpg", img_flip)
        if save:
            cv2.imwrite(config.STANF_CONV + fileName, img)
        if img is None:
            print(f"None! {fileName}")
        else:
            img = img.reshape((config.Image_size, config.Image_size, 1))
            lijst.append(img)
    return lijst


def readConvImages(imgs):
    lijst = []
    for fileName in imgs:
        img = cv2.imread(config.STANF_CONV + fileName, 0)
        img2 = cv2.imread(config.STANF_CONV + fileName[:-4] + "_flip.jpg", 0)
        if img is None or img2 is None:
            print(f"None! {fileName}")
        else:
            img = img.reshape((config.Image_size, config.Image_size, 1))
            img2 = img2.reshape((config.Image_size, config.Image_size, 1))
            lijst.append(img)
            lijst.append(img2)
    return lijst


def removeNumberAndExt(img: str):
    return img[:-7]


def splitTrain(files, labels):
    if len(files) != len(labels):
        print("ERROR! Files and labels don't have the same length")
        return
    train_imgs, val_imgs, train_labels, val_labels = [], [], [], []
    temp = []
    prev_lab = removeNumberAndExt(files[0])
    files.append("END_000.jpg")
    for i in range(0, len(files)):
        lab = removeNumberAndExt(files[i])
        if lab != prev_lab:
            aantalTrain = len(temp) - int(len(temp) * config.Validate_perc)
            # print(f"Length for lab: {len(temp)} and number in Train: {aantalTrain}")
            for j in temp[:aantalTrain]:
                train_imgs.append(files[j])
                train_labels.append(labels[j])
            for j in temp[aantalTrain:]:
                val_imgs.append(files[j])
                val_labels.append(labels[j])

            # for new label
            prev_lab = lab
            temp = [i]
        else:
            temp.append(i)
    return train_imgs, np.array(train_labels), val_imgs, np.array(val_labels)


def fit_model(model, train_images: np.ndarray, train_labels: np.ndarray, val_images: np.ndarray, val_labels: np.ndarray,
              model_name: str, printing: bool = False):
    history = model.fit(train_images, train_labels, epochs=config.Epochs,
                        validation_data=(val_images, val_labels),
                        batch_size=config.Batch_size)
    if printing:
        evaluate_model(model, val_images, val_labels)

    return history, model


def evaluate_model(model, test_images: np.ndarray, test_labels: np.ndarray):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
    print(test_acc)


def test_model(model, test_images: np.ndarray, test_labels: np.ndarray):
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Acc: {test_acc} & loss: {test_loss}")
    return test_loss, test_acc


def getUniques(labels):
    used = set()
    unique = [x for x in labels if x not in used and (used.add(x) or True)]
    dict = {}
    for i in range(0, len(unique)):
        dict[unique[i]] = np.uint8(i)
    return unique, dict


def double_labels(labs):
    list = []
    for i in labs:
        list.append(i)
        list.append(i)
    return list


def main():
    stf_train_files, stf_train_labels_S, stf_test_files, stf_test_labels = train_test_stanford(False)
    input_shape = (112, 112, 1)
    uniqueLabels, dictionary = getUniques(stf_test_labels)
    stf_train_labels_ind = [dictionary[lab] for lab in stf_train_labels_S]
    stf_test_labels_ind = [dictionary[lab] for lab in stf_test_labels]
    # run once
    # preprocess(stf_train_files, True)  # True als je ze wilt opslaan
    # preprocess(stf_test_files, True)  # True als je ze wilt opslaan

    if config.Use_converted:
        stf_train_imgs = np.array(readConvImages(stf_train_files))
        stf_test_imgs = np.array(readConvImages(stf_test_files))

    stf_train_labels = np.array(double_labels(stf_train_labels_ind))
    stf_test_labels = np.array(double_labels(stf_test_labels_ind))

    stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stf_train_imgs,
                                                                                      stf_train_labels,
                                                                                      test_size=config.Validate_perc,
                                                                                      stratify=stf_train_labels)
    # grid search grid
    activation1 = ['relu', 'sigmoid', 'tanh']
    activation2 = ['relu', 'sigmoid', 'tanh']
    activation3 = ['relu', 'sigmoid', 'tanh', 'softmax']
    dropout = [0.5, 0.2, 0.0]
    optimizer = ['adam']
    hidden_layers = [1, 2]
    hidden_layer_neurons = [60, 80, 100]
    filter_size = [8, 16, 32]
    kernel_size = [3, 5]
    conv_layers = [0, 1]
    hyperparameters = dict(optimizer=optimizer, activation1=activation1, activation2=activation2,
                           activation3=activation3, dropout=dropout,
                           hidden_layers=hidden_layers, hidden_layer_neurons=hidden_layer_neurons,
                           filter_size=filter_size, kernel_size=kernel_size,
                           conv_layers=conv_layers, input_shape=[input_shape])

    images_train = np.concatenate([stf_train_imgs, stf_val_imgs])
    labels_train = np.concatenate([stf_train_labels, stf_val_labels])
    test_fold = [-1] * len(stf_train_imgs) + [0] * len(stf_val_imgs)

    my_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=make_baseline_model)
    grid = RandomizedSearchCV(estimator=my_classifier, param_distributions=hyperparameters, verbose=5,
                              cv=StratifiedShuffleSplit(1), n_iter=15)
    print("Start fitting")
    # model_result = model.fit(images_train, labels_train, epochs=config.Epochs,
    #                        validation_data=(stf_val_imgs, stf_val_labels),
    #                        batch_size=config.Batch_size)

    grid_result = grid.fit(images_train, labels_train, epochs=config.Epochs,
                           validation_data=(stf_val_imgs, stf_val_labels),
                           batch_size=config.Batch_size, verbose=5)
    print(test_model(grid_result.best_estimator_, stf_test_imgs, stf_test_labels))

    print(grid_result.best_params_)
    # history, model = fit_model(model, stf_train_imgs, stf_train_labels, stf_val_imgs, stf_val_labels, "base",
    #                          printing=True)
    print("Done")


if __name__ == '__main__':
    main()
