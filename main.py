import os

import cv2
import joblib
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from tensorflow.keras import layers, models

import HelperFunctions as HF
import OpticalFlow as OptF
import config
import data
import fusion


def transfer_learn_model(model_path, new_output_layer, freeze: bool = False, lr=0.0001):
    old_model = models.load_model(model_path)
    model = models.Sequential()

    for layer in old_model.layers[:-1]:
        model.add(layer)

    if freeze:
        for layer in model.layers:
            layer.trainable = False

    model.add(new_output_layer)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr
    )
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer,
                  metrics=['accuracy'])
    print(model.summary())
    return model


def make_baseline_model(input_shape, hidden_layers, activation1='relu', activation2='relu', activation3='relu',
                        optimizer='adam', conv_layers=2, filter_size=16, kernel_size=3, dropout=0.0, output_size=40,
                        filter_multiplier=1, dubbel_conv: bool = True, k_reg=None):
    model = models.Sequential()
    if kernel_size == 5 and conv_layers == 3:
        conv_layers = 2

    if dubbel_conv:
        model.add(layers.Conv2D(filter_size, (3, 3), activation=activation1, input_shape=input_shape,
                                kernel_regularizer=k_reg))
        model.add(layers.Conv2D(filter_size, (3, 3), activation=activation1, kernel_regularizer=k_reg))  # 108
    else:
        model.add(layers.Conv2D(filter_size, (5, 5), activation=activation1, input_shape=input_shape,
                                kernel_regularizer=k_reg))
    model.add(layers.MaxPooling2D((2, 2)))  # 54
    filter_size = int(filter_size * filter_multiplier)
    for i in range(conv_layers):
        model.add(layers.Conv2D(filter_size * filter_multiplier, (kernel_size, kernel_size), activation=activation1,
                                kernel_regularizer=k_reg))
        model.add(layers.MaxPooling2D((2, 2)))
        filter_size = int(filter_size * filter_multiplier)
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg))
    model.add(layers.Flatten())
    model.add(layers.Dense(hidden_layers, activation=activation2, kernel_regularizer=k_reg))
    model.add(layers.Dense(hidden_layers, activation=activation2, kernel_regularizer=k_reg))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_size, activation=activation3))
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())
    return model


def cyclical_learning_rate(initial_learning_rate, maximal_learning_rate, step_size, scale_fn, scale_mode,
                           type="triangle2"):
    if type == "cyclical":
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                  maximal_learning_rate=maximal_learning_rate,
                                                  step_size=step_size,
                                                  scale_fn=scale_fn,
                                                  scale_mode=scale_mode)
    elif type == "exponential":
        clr = tfa.optimizers.ExponentialCyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                             maximal_learning_rate=maximal_learning_rate,
                                                             step_size=step_size,
                                                             scale_fn=scale_fn,
                                                             scale_mode=scale_mode)
    elif type == "triangle":
        clr = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                            maximal_learning_rate=maximal_learning_rate,
                                                            step_size=step_size,
                                                            scale_fn=scale_fn,
                                                            scale_mode=scale_mode)
    elif type == "triangle2":
        clr = tfa.optimizers.Triangular2CyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                             maximal_learning_rate=maximal_learning_rate,
                                                             step_size=step_size,
                                                             scale_fn=scale_fn,
                                                             scale_mode=scale_mode)
    else:
        print("Wrong cyclical learn rate type supplied, resorting to default")
        clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                  maximal_learning_rate=maximal_learning_rate,
                                                  step_size=step_size,
                                                  scale_fn=scale_fn,
                                                  scale_mode=scale_mode)

    return clr


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


def video_name_to_image_name(testx, testy, trainx, trainy):
    # very unoptimized, but it works :)
    for i in range(len(testx)):
        name = testx[i]
        name = name[:-4] + ".jpg"
        testx[i] = name
    for i in range(len(testy)):
        name = testy[i]
        name = name[:-4] + ".jpg"
        testy[i] = name
    for i in range(len(trainx)):
        name = trainx[i]
        name = name[:-4] + ".jpg"
        trainx[i] = name
    for i in range(len(trainy)):
        name = trainy[i]
        name = name[:-4] + ".jpg"
        trainy[i] = name
    return testx, testy, trainx, trainy


def testOpticalFlow():
    tv_test_vid, tv_test_label, tv_tr_v, tv_tr_l = data.train_tests_tv(True)
    tv_tr_l = HF.convertLabel(tv_tr_l)

    # Iets met de shape ofzo

    aantal_frames = 10
    input_shape = (config.Image_size, config.Image_size, aantal_frames * 2)
    print("Make model")
    kernel_regulariser = tf.keras.regularizers.l2(0.01)
    model = make_baseline_model(input_shape, hidden_layer_neurons=60, activation3='softmax', output_size=4,
                                k_reg=kernel_regulariser)

    print("Get flow data")
    flow_data = OptF.getVideosFlow(tv_tr_v, config.TV_VIDEOS_SLASH, True, config.Image_size, aantal_frames)
    tv_train, tv_val, tv_train_l, tv_val_l = train_test_split(flow_data, tv_tr_l, test_size=0.15, stratify=tv_tr_l)

    tv_train_l, tv_val_l = np.array(tv_train_l), np.array(tv_val_l)
    print(type(tv_train))
    print(type(tv_train_l))
    print(type(tv_val))
    print(type(tv_val_l))
    print(f"val: {tv_val_l[0]}")
    print(tv_train.shape)
    print(tv_train_l.shape)
    input()
    model_result = model.fit(tv_train, tv_train_l, epochs=config.Epochs, validation_data=(tv_val, tv_val_l),
                             batch_size=config.Batch_size)
    print("Done!")
    input()


def random_search(input_shape, newImgs, labss, testImgs, testLabs, n_iter=45):
    # grid search grid
    activation1 = ['relu']
    activation2 = ['relu']
    activation3 = ['softmax']
    dropout = [0.5, 0.2, 0.0]
    optimizer = ['adam']
    hidden_layers = [60]
    hidden_layers2 = [[60], [40], [80], [60, 60], [60, 40], [80, 80], [80, 60], [80, 40]]

    filter_size = [8, 16]
    filter_multiplier = [1, 1.5, 2]
    kernel_size = [3, 5]
    conv_layers = [1, 2, 3]
    dubbel_conv = [True, False]
    hyperparameters = dict(optimizer=optimizer, activation1=activation1, activation2=activation2,
                           activation3=activation3, dropout=dropout,
                           hidden_layers=hidden_layers, filter_size=filter_size, kernel_size=kernel_size,
                           conv_layers=conv_layers, input_shape=[input_shape], filter_multiplier=filter_multiplier,
                           dubbel_conv=dubbel_conv)

    my_classifier = tf.keras.wrappers.scikit_learn.KerasClassifier(build_fn=make_baseline_model)
    grid = RandomizedSearchCV(estimator=my_classifier, param_distributions=hyperparameters, verbose=1,
                              cv=StratifiedShuffleSplit(1), n_iter=n_iter, refit=True)

    grid_result = grid.fit(newImgs, labss, epochs=config.Epochs,
                           validation_data=(testImgs, testLabs),
                           batch_size=config.Batch_size, verbose=1)
    # joblib.dump(grid, 'random_search.pkl')
    # joblib.load('random_search.pkl')
    print("Done")
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print(" ")
    print("All fits: ")
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    _, _ = test_model(grid.best_estimator_, testImgs, testLabs)

    # history, model = fit_model(model, stf_train_imgs, stf_train_labels, stf_val_imgs, stf_val_labels, "base",
    #                          printing=True)
    input("press any key to continue...")


# Testing image augmentation
def test():
    seq = iaa.Sequential([
        iaa.TranslateX(px=(-20, 20), mode='reflect'),
        iaa.GaussianBlur(sigma=(0, 1.5))
    ], random_order=True).to_deterministic()

    blur = iaa.GaussianBlur(sigma=(0, 1.5)).to_deterministic()
    newImgs, labss = HF.getDataSet(["applauding_004.jpg"], config.STANF_CONV_CROP, False, [1])
    applauding = cv2.imread(config.STANF_CONV_CROP + "applauding_004.jpg", 1)
    img_blur = blur(image=applauding)

    # img = seq(images=np.array([applauding]))
    # print(type(img))
    # print(img.shape)
    cv2.imshow("img", img_blur)
    cv2.waitKey(0)
    # for i in img:
    #     print(type(i))
    #     # print(i)
    #     cv2.imshow("img", i)
    #     cv2.waitKey(0)
    print("DONE")
    cv2.destroyAllWindows()


def makeModelsFinal(stanf_train, stanf_train_lab, stanf_test, stanf_test_lab, input_shape, kernel_reg):
    # Make first model
    print("Making the first model...")
    # Variabelen moeten wel aangepast worden waarschijnlijk
    model_base = make_baseline_model(input_shape, conv_layers=3, hidden_layer_neurons=60, activation3='softmax',
                                     kernel_reg=kernel_reg)
    hist_base, model_base = model_base.fit(stanf_train, stanf_train_lab, validation_data=(stanf_test, stanf_test_lab),
                                           epochs=config.Epochs, batch_size=config.Batch_size)
    test_loss_base, test_acc_base = model_base.evaluate(stanf_test, stanf_test_lab)
    print(f"Test acc: {test_acc_base} & loss: {test_loss_base}")

    # Plot test loss and accuracy

    # Make second model, based on transfer learning
    print("Making the second model...")
    # transferlearn model
    # Moet nog ingevuld worden
    # model2 = ...
    # Plot test loss and accuracy

    # Make third model, etc..
    print("Making the third model...")
    """
    WAARDEN MOETEN GELIJK ZIJN AAN MODEL1, MAAR MAG NIET GETRAIND ZIJN VOLGENS MIJ.
    """
    aantal_frames = 10
    input_shape3 = (config.Image_size, config.Image_size, aantal_frames * 2)
    model3 = make_baseline_model(input_shape3, conv_layers=3, hidden_layer_neurons=60, activation3='softmax')
    # ZONDER AUGMENTATION
    flow_train, flow_train_l, flow_test, flow_test_l = data.getFusionData(aantal_frames, augm=False, extra_fusion=False)
    # MET AUGMENTATION EN EXTRA DATA, ANDERE FRAMES GESELECTEERD
    # flow_train, flow_train_l, flow_test, flow_test_l = data.getFusionData(aantal_frames, augm=True, extra_fusion=True)

    hist_model3, model3 = model3.fit(flow_train, flow_train_l, epochs=config.Epochs, batch_size=config.Batch_size)
    test_loss_m3, test_acc_m3 = model3.evaluate(flow_test, flow_test_l)
    # Plot test loss and accuracy

    """Make fourth model, etc.."""
    print("Making the fourth model...")
    # model4 = fusion.standardModel4(model2, model3, True, kernel_reg)

    # Alternatief op basis van Assignment 5 Q&A
    # model4 = fusion.standardModel4(model3, model2, True, kernel_reg)

    # train_data_model4 = [data_2, data_3]
    # labels_model4 = .... zou hetzelfde moeten zijn als voor model2 en model3
    # model4.fit(train_data_model4, labels_model4, epochs=config.Epochs)

    # Plot test loss and accuracy

    print()


def main():
    input_shape = (config.Image_size, config.Image_size, 3)
    # Test Optical Flow
    # testOpticalFlow()

    # stf_trainset: whole training set (incl. val); stf_train_imgs: only training (excl val)
    print("Get DataSet")
    stf_trainset, stf_trainset_lab, stf_test, stf_test_lab = data.getStanfordData()
    stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stf_trainset, stf_trainset_lab,
                                                                                      test_size=config.Validate_perc)

    # Function to train and make the four models
    # makeModelsFinal(stf_trainset, stf_trainset_lab, stf_test, stf_test_lab, input_shape)

    # Test the different forms of augmentation
    # test()

    # Choice task cyclical learning rate scheduler
    clr = cyclical_learning_rate(1e-4, 1e-3, step_size=10, scale_fn=lambda x: 1, scale_mode='cycle', type="cyclical")
    print("start random search")
    random_search(input_shape, stf_trainset, stf_trainset_lab, stf_test, stf_test_lab, n_iter=10)

    print("Make model")
    # model = make_baseline_model(input_shape, conv_layers=3, hidden_layer_neurons=60, activation3='softmax',
    #                             k_reg=tf.keras.regularizers.l2(0.01))
    # model_result = model.fit(stf_train_imgs, stf_train_labels, epochs=config.Epochs,
    #                          validation_data=(stf_val_imgs, stf_val_labels), batch_size=config.Batch_size)

    if not os.path.isfile(config.MODELS + "Baseline.h5"):
        print("Model file not found, creating...")
        model = make_baseline_model(input_shape, hidden_layer_neurons=60, activation3='softmax', conv_layers=3)
        # model_result = model.fit(stf_train_imgs, stf_train_labels, epochs=config.Epochs,
        #                          validation_data=(stf_val_imgs, stf_val_labels), batch_size=config.Batch_size)
        model_result = model.fit(stf_trainset, stf_trainset_lab, validation_data=(stf_test, stf_test_lab),
                                 epochs=config.Epochs, batch_size=config.Batch_size)
        model.save(config.MODELS + "Baseline.h5")
    else:
        print("Model file located")
        model = models.load_model(config.MODELS + "Baseline.h5")

    test_loss, test_acc = test_model(model, stf_test, stf_test_lab)

    # Transfer learn to TV-HI data
    HF.take_middle_frame(config.TV_VIDEOS)
    tv_test_files, tv_test_labels, tv_train_files, tv_train_labels = data.train_tests_tv()

    tv_test_files, tv_test_labels, tv_train_files, tv_train_labels = video_name_to_image_name(tv_test_files,
                                                                                              tv_test_labels,
                                                                                              tv_train_files,
                                                                                              tv_train_labels)
    uniqueLabels, dictionary = HF.getUniques(tv_test_labels)
    tv_train_labels_ind = [dictionary[lab] for lab in tv_train_labels]
    tv_test_labels_ind = [dictionary[lab] for lab in tv_test_labels]

    # Run this once
    # HF.convertAndCropImg(tv_train_files, config.TV_IMG, True, True, config.Image_size, config.TV_CONV_CROP)
    # HF.convertAndCropImg(tv_test_files, config.TV_IMG, True, True, config.Image_size, config.TV_CONV_CROP)
    # HF.convertNew(tv_test_files, config.TV_IMG, config.Image_size, config.TV_CONV, config.TV_CONV_CROP)

    newImgs, labss = HF.getDataSet(tv_train_files, config.TV_CONV_CROP, False, tv_train_labels_ind)
    tv_train_imgs, tv_val_imgs, tv_train_labels, tv_val_labels = train_test_split(newImgs, labss,
                                                                                  test_size=config.Validate_perc)

    if not os.path.isfile(config.MODELS + "TV_TL.h5"):
        print("Model file not found, creating...")
        tv_output_layer = layers.Dense(4, activation="softmax", name="Dense_output")
        tl_model = transfer_learn_model(config.MODELS + "Baseline.h5", tv_output_layer, freeze=True, lr=clr)
        tl_model_result = tl_model.fit(tv_train_imgs, tv_train_labels, epochs=config.Epochs,
                                       validation_data=(tv_val_imgs, tv_val_labels), batch_size=config.Batch_size)
        tl_model.save(config.MODELS + "TV_TL.h5")
    else:
        print("Model file located")
        tl_model = models.load_model(config.MODELS + "TV_TL.h5")

    print("Do you want to start grid search? Press any key")
    input()

    # Run once to get the cropped images
    # HF.convertAndCropImg(stf_train_files, True, True, config.Image_size, config.STANF_CONV_CROP)
    # HF.convertAndCropImg(stf_test_files, True, True, config.Image_size, config.STANF_CONV_CROP)
    # HF.convertNew(stf_test_files, config.Image_size, config.STANF_CONV, config.STANF_CONV_CROP)
    # input()
    # if config.Use_converted:
    #     cropped_ = True
    #     stf_train_imgs = np.array(readConvImages(stf_train_files, cropped=cropped_, grayScale=False))
    #     stf_test_imgs = np.array(readConvImages(stf_test_files, cropped=cropped_, grayScale=False))
    #
    # stf_train_labels = np.array(HF.double_labels(stf_train_labels_ind))
    # stf_test_labels = np.array(HF.double_labels(stf_test_labels_ind))
    #
    # stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stf_train_imgs,
    #                                                                                   stf_train_labels,
    #                                                                                   test_size=config.Validate_perc,
    #                                                                                   stratify=stf_train_labels)
    #
    # images_train = np.concatenate([stf_train_imgs, stf_val_imgs])
    # labels_train = np.concatenate([stf_train_labels, stf_val_labels])
    # test_fold = [-1] * len(stf_train_imgs) + [0] * len(stf_val_imgs)
    # print("Start fitting")
    # model = make_baseline_model(input_shape, conv_layers=3, hidden_layer_neurons=60, activation3='softmax')
    # # conv = 3, neurons = 60: acc. 0.10, val_acc. 0.0838 < RANDOM, NIET REPLICEERBAAR
    # model_result = model.fit(stf_train_imgs, stf_train_labels, epochs=config.Epochs,
    #                          validation_data=(stf_val_imgs, stf_val_labels), batch_size=config.Batch_size)

    input()


if __name__ == '__main__':
    main()

"""Using lot of data augmentation: augmentImages(imgs, False, True, False, True, True, True, True)
model = make_baseline_model(input_shape, conv_layers=3, hidden_layer_neurons=60, activation3='softmax')
Epoch 1/40
675/675 [==============================] - 199s 294ms/step - loss: 3.9538 - accuracy: 0.0276 - val_loss: 3.6581 - val_accuracy: 0.0433
Epoch 2/40
675/675 [==============================] - 187s 278ms/step - loss: 3.6228 - accuracy: 0.0536 - val_loss: 3.4603 - val_accuracy: 0.0763
Epoch 3/40
675/675 [==============================] - 187s 276ms/step - loss: 3.3902 - accuracy: 0.1006 - val_loss: 3.2722 - val_accuracy: 0.1187
Epoch 4/40
675/675 [==============================] - 189s 280ms/step - loss: 3.1615 - accuracy: 0.1485 - val_loss: 3.1475 - val_accuracy: 0.1433
Epoch 5/40
675/675 [==============================] - 183s 271ms/step - loss: 2.9319 - accuracy: 0.2007 - val_loss: 2.9946 - val_accuracy: 0.2008
Epoch 6/40
675/675 [==============================] - 189s 280ms/step - loss: 2.6977 - accuracy: 0.2587 - val_loss: 2.8129 - val_accuracy: 0.2342
Epoch 7/40
675/675 [==============================] - 190s 281ms/step - loss: 2.4774 - accuracy: 0.3130 - val_loss: 2.7282 - val_accuracy: 0.2671
Epoch 8/40
675/675 [==============================] - 184s 272ms/step - loss: 2.2970 - accuracy: 0.3634 - val_loss: 2.6304 - val_accuracy: 0.2946
Epoch 9/40
675/675 [==============================] - 192s 284ms/step - loss: 2.0908 - accuracy: 0.4210 - val_loss: 2.5496 - val_accuracy: 0.3258
Epoch 10/40
675/675 [==============================] - 189s 280ms/step - loss: 1.9522 - accuracy: 0.4574 - val_loss: 2.5584 - val_accuracy: 0.3304
Epoch 11/40
675/675 [==============================] - 190s 281ms/step - loss: 1.8168 - accuracy: 0.4912 - val_loss: 2.4327 - val_accuracy: 0.3704
Epoch 12/40
675/675 [==============================] - 189s 280ms/step - loss: 1.6894 - accuracy: 0.5255 - val_loss: 2.3967 - val_accuracy: 0.3787
Epoch 13/40
675/675 [==============================] - 189s 280ms/step - loss: 1.6183 - accuracy: 0.5487 - val_loss: 2.3859 - val_accuracy: 0.4025
Epoch 14/40
675/675 [==============================] - 184s 273ms/step - loss: 1.5371 - accuracy: 0.5687 - val_loss: 2.2897 - val_accuracy: 0.4392
Epoch 15/40
675/675 [==============================] - 185s 273ms/step - loss: 1.4429 - accuracy: 0.5871 - val_loss: 2.3492 - val_accuracy: 0.4133
Epoch 16/40
675/675 [==============================] - 187s 278ms/step - loss: 1.3934 - accuracy: 0.5988 - val_loss: 2.3513 - val_accuracy: 0.4300
Epoch 17/40
675/675 [==============================] - 189s 280ms/step - loss: 1.3182 - accuracy: 0.6254 - val_loss: 2.3603 - val_accuracy: 0.4367
Epoch 18/40
675/675 [==============================] - 193s 286ms/step - loss: 1.2478 - accuracy: 0.6458 - val_loss: 2.2809 - val_accuracy: 0.4608
Epoch 19/40
675/675 [==============================] - 185s 274ms/step - loss: 1.2261 - accuracy: 0.6437 - val_loss: 2.3980 - val_accuracy: 0.4429
Epoch 20/40
675/675 [==============================] - 186s 275ms/step - loss: 1.1751 - accuracy: 0.6585 - val_loss: 2.4859 - val_accuracy: 0.4275
"""

# OUD
"""
model = make_baseline_model(input_shape, conv_layers=3, hidden_layers=0, hidden_layer_neurons=50)
Epoch 1/40
225/225 [==============================] - 32s 139ms/step - loss: 8.4073 - accuracy: 0.0202 - val_loss: 3.6886 - val_accuracy: 0.0312
Epoch 2/40
225/225 [==============================] - 29s 131ms/step - loss: 3.6884 - accuracy: 0.0273 - val_loss: 3.6886 - val_accuracy: 0.0312
Epoch 3/40
225/225 [==============================] - 29s 131ms/step - loss: 3.6868 - accuracy: 0.0272 - val_loss: 3.6876 - val_accuracy: 0.0325
Epoch 4/40
225/225 [==============================] - 29s 130ms/step - loss: 3.6818 - accuracy: 0.0369 - val_loss: 3.6813 - val_accuracy: 0.0338
Epoch 5/40
225/225 [==============================] - 30s 133ms/step - loss: 3.6764 - accuracy: 0.0369 - val_loss: 3.6826 - val_accuracy: 0.0375
Epoch 6/40
225/225 [==============================] - 30s 131ms/step - loss: 3.6644 - accuracy: 0.0392 - val_loss: 3.6767 - val_accuracy: 0.0400
Epoch 7/40
225/225 [==============================] - 30s 134ms/step - loss: 3.6575 - accuracy: 0.0501 - val_loss: 3.6770 - val_accuracy: 0.0362
Epoch 8/40
225/225 [==============================] - 30s 131ms/step - loss: 3.6551 - accuracy: 0.0501 - val_loss: 3.6745 - val_accuracy: 0.0362
Epoch 9/40
225/225 [==============================] - 31s 139ms/step - loss: 3.6424 - accuracy: 0.0560 - val_loss: 3.6766 - val_accuracy: 0.0413
Epoch 10/40
225/225 [==============================] - 30s 131ms/step - loss: 3.6290 - accuracy: 0.0591 - val_loss: 3.6828 - val_accuracy: 0.0425
Epoch 11/40
225/225 [==============================] - 30s 134ms/step - loss: 3.6140 - accuracy: 0.0613 - val_loss: 3.6757 - val_accuracy: 0.0375
Epoch 12/40
225/225 [==============================] - 29s 130ms/step - loss: 3.5918 - accuracy: 0.0709 - val_loss: 3.6717 - val_accuracy: 0.0388
Epoch 13/40
225/225 [==============================] - 29s 130ms/step - loss: 3.5781 - accuracy: 0.0747 - val_loss: 3.7019 - val_accuracy: 0.0413
Epoch 14/40
225/225 [==============================] - 29s 129ms/step - loss: 3.5668 - accuracy: 0.0862 - val_loss: 3.6654 - val_accuracy: 0.0413
Epoch 15/40
225/225 [==============================] - 29s 129ms/step - loss: 3.5292 - accuracy: 0.0995 - val_loss: 3.6825 - val_accuracy: 0.0538
Epoch 16/40
225/225 [==============================] - 29s 130ms/step - loss: 3.5111 - accuracy: 0.1017 - val_loss: 3.6665 - val_accuracy: 0.0562
Epoch 17/40
225/225 [==============================] - 30s 131ms/step - loss: 3.4900 - accuracy: 0.1054 - val_loss: 3.6774 - val_accuracy: 0.0463
Epoch 18/40
225/225 [==============================] - 29s 130ms/step - loss: 3.4600 - accuracy: 0.1173 - val_loss: 3.6814 - val_accuracy: 0.0550
Epoch 19/40
225/225 [==============================] - 29s 128ms/step - loss: 3.4494 - accuracy: 0.1160 - val_loss: 3.6989 - val_accuracy: 0.0538
Epoch 20/40
225/225 [==============================] - 29s 130ms/step - loss: 3.3921 - accuracy: 0.1321 - val_loss: 3.7093 - val_accuracy: 0.0512
Epoch 21/40
225/225 [==============================] - 29s 129ms/step - loss: 3.4120 - accuracy: 0.1257 - val_loss: 3.6852 - val_accuracy: 0.0650
Epoch 22/40
225/225 [==============================] - 29s 130ms/step - loss: 3.3429 - accuracy: 0.1485 - val_loss: 3.7594 - val_accuracy: 0.0688
Epoch 23/40
225/225 [==============================] - 29s 130ms/step - loss: 3.3382 - accuracy: 0.1553 - val_loss: 3.7070 - val_accuracy: 0.0625
Epoch 24/40
225/225 [==============================] - 29s 130ms/step - loss: 3.2844 - accuracy: 0.1641 - val_loss: 3.7194 - val_accuracy: 0.0725
Epoch 25/40
225/225 [==============================] - 29s 130ms/step - loss: 3.2309 - accuracy: 0.1768 - val_loss: 3.7003 - val_accuracy: 0.0650
Epoch 26/40
225/225 [==============================] - 29s 129ms/step - loss: 3.1842 - accuracy: 0.1917 - val_loss: 3.7584 - val_accuracy: 0.0637
Epoch 27/40
225/225 [==============================] - 29s 128ms/step - loss: 3.1915 - accuracy: 0.1920 - val_loss: 3.7197 - val_accuracy: 0.0750
Epoch 28/40
225/225 [==============================] - 29s 129ms/step - loss: 3.1861 - accuracy: 0.1949 - val_loss: 3.7509 - val_accuracy: 0.0712
Epoch 29/40
225/225 [==============================] - 29s 130ms/step - loss: 3.1346 - accuracy: 0.2045 - val_loss: 3.7285 - val_accuracy: 0.0688
Epoch 30/40
225/225 [==============================] - 29s 130ms/step - loss: 3.1148 - accuracy: 0.2180 - val_loss: 3.8844 - val_accuracy: 0.0712
Epoch 31/40
225/225 [==============================] - 29s 130ms/step - loss: 3.0633 - accuracy: 0.2289 - val_loss: 3.7800 - val_accuracy: 0.0737
Epoch 32/40
225/225 [==============================] - 29s 129ms/step - loss: 3.0377 - accuracy: 0.2394 - val_loss: 3.8555 - val_accuracy: 0.0637
Epoch 33/40
225/225 [==============================] - 29s 129ms/step - loss: 2.9875 - accuracy: 0.2513 - val_loss: 3.8345 - val_accuracy: 0.0725
Epoch 34/40
225/225 [==============================] - 30s 132ms/step - loss: 2.9550 - accuracy: 0.2565 - val_loss: 3.9102 - val_accuracy: 0.0725
Epoch 35/40
225/225 [==============================] - 29s 130ms/step - loss: 2.9530 - accuracy: 0.2576 - val_loss: 3.8975 - val_accuracy: 0.0637
Epoch 36/40
225/225 [==============================] - 29s 129ms/step - loss: 2.9369 - accuracy: 0.2619 - val_loss: 3.8912 - val_accuracy: 0.0688
Epoch 37/40
225/225 [==============================] - 29s 130ms/step - loss: 2.8990 - accuracy: 0.2658 - val_loss: 3.9415 - val_accuracy: 0.0775
Epoch 38/40
225/225 [==============================] - 30s 132ms/step - loss: 2.8800 - accuracy: 0.2761 - val_loss: 3.9983 - val_accuracy: 0.0800
Epoch 39/40
225/225 [==============================] - 29s 128ms/step - loss: 2.8149 - accuracy: 0.2918 - val_loss: 3.9683 - val_accuracy: 0.0763
Epoch 40/40
225/225 [==============================] - 29s 130ms/step - loss: 2.7668 - accuracy: 0.3057 - val_loss: 4.0032 - val_accuracy: 0.0737
"""

"""
model = make_baseline_model(input_shape, conv_layers=3,  hidden_layer_neurons=50)
Epoch 1/40
225/225 [==============================] - 29s 127ms/step - loss: 4.4177 - accuracy: 0.0248 - val_loss: 3.6912 - val_accuracy: 0.0237
Epoch 2/40
225/225 [==============================] - 26s 118ms/step - loss: 3.6853 - accuracy: 0.0313 - val_loss: 3.6827 - val_accuracy: 0.0325
Epoch 3/40
225/225 [==============================] - 29s 130ms/step - loss: 3.6837 - accuracy: 0.0346 - val_loss: 3.6853 - val_accuracy: 0.0237
Epoch 4/40
225/225 [==============================] - 29s 130ms/step - loss: 3.6833 - accuracy: 0.0357 - val_loss: 3.6846 - val_accuracy: 0.0300
Epoch 5/40
225/225 [==============================] - 30s 135ms/step - loss: 3.6763 - accuracy: 0.0425 - val_loss: 3.6863 - val_accuracy: 0.0300
Epoch 6/40
225/225 [==============================] - 28s 123ms/step - loss: 3.6645 - accuracy: 0.0474 - val_loss: 3.6787 - val_accuracy: 0.0362
Epoch 7/40
225/225 [==============================] - 25s 112ms/step - loss: 3.6364 - accuracy: 0.0591 - val_loss: 3.6564 - val_accuracy: 0.0400
Epoch 8/40
225/225 [==============================] - 25s 112ms/step - loss: 3.6176 - accuracy: 0.0686 - val_loss: 3.6457 - val_accuracy: 0.0625
Epoch 9/40
225/225 [==============================] - 25s 112ms/step - loss: 3.5878 - accuracy: 0.0721 - val_loss: 3.6477 - val_accuracy: 0.0512
Epoch 10/40
225/225 [==============================] - 25s 112ms/step - loss: 3.5485 - accuracy: 0.0872 - val_loss: 3.6644 - val_accuracy: 0.0575
Epoch 11/40
225/225 [==============================] - 25s 112ms/step - loss: 3.5142 - accuracy: 0.1041 - val_loss: 3.6172 - val_accuracy: 0.0800
Epoch 12/40
225/225 [==============================] - 25s 112ms/step - loss: 3.4435 - accuracy: 0.1218 - val_loss: 3.5574 - val_accuracy: 0.1037
Epoch 13/40
225/225 [==============================] - 25s 111ms/step - loss: 3.4163 - accuracy: 0.1334 - val_loss: 3.6068 - val_accuracy: 0.0900
Epoch 14/40
225/225 [==============================] - 25s 111ms/step - loss: 3.3379 - accuracy: 0.1438 - val_loss: 3.5991 - val_accuracy: 0.0862
Epoch 15/40
225/225 [==============================] - 25s 111ms/step - loss: 3.2928 - accuracy: 0.1716 - val_loss: 3.6490 - val_accuracy: 0.0825
Epoch 16/40
225/225 [==============================] - 25s 112ms/step - loss: 3.2476 - accuracy: 0.1772 - val_loss: 3.6616 - val_accuracy: 0.0862
Epoch 17/40
225/225 [==============================] - 25s 111ms/step - loss: 3.1806 - accuracy: 0.1999 - val_loss: 3.6978 - val_accuracy: 0.0688
Epoch 18/40
225/225 [==============================] - 25s 111ms/step - loss: 3.1197 - accuracy: 0.2233 - val_loss: 3.7279 - val_accuracy: 0.0763
Epoch 19/40
225/225 [==============================] - 25s 111ms/step - loss: 3.0553 - accuracy: 0.2356 - val_loss: 3.7165 - val_accuracy: 0.0913
Epoch 20/40
225/225 [==============================] - 25s 111ms/step - loss: 3.0214 - accuracy: 0.2440 - val_loss: 3.7646 - val_accuracy: 0.0800
Epoch 21/40
225/225 [==============================] - 25s 111ms/step - loss: 2.9662 - accuracy: 0.2582 - val_loss: 3.8299 - val_accuracy: 0.0750
Epoch 22/40
225/225 [==============================] - 25s 111ms/step - loss: 2.8989 - accuracy: 0.2778 - val_loss: 3.7975 - val_accuracy: 0.0900
Epoch 23/40
225/225 [==============================] - 25s 111ms/step - loss: 2.8890 - accuracy: 0.2869 - val_loss: 3.8426 - val_accuracy: 0.0875
Epoch 24/40
225/225 [==============================] - 25s 111ms/step - loss: 2.8086 - accuracy: 0.3056 - val_loss: 3.9162 - val_accuracy: 0.0925
Epoch 25/40
225/225 [==============================] - 25s 111ms/step - loss: 2.7271 - accuracy: 0.3231 - val_loss: 3.9207 - val_accuracy: 0.0763
Epoch 26/40
225/225 [==============================] - 25s 111ms/step - loss: 2.6941 - accuracy: 0.3296 - val_loss: 3.8937 - val_accuracy: 0.1013
Epoch 27/40
225/225 [==============================] - 25s 111ms/step - loss: 2.7327 - accuracy: 0.3302 - val_loss: 3.9394 - val_accuracy: 0.0862
Epoch 28/40
225/225 [==============================] - 25s 113ms/step - loss: 2.6996 - accuracy: 0.3315 - val_loss: 4.0955 - val_accuracy: 0.0800
Epoch 29/40
225/225 [==============================] - 25s 111ms/step - loss: 2.6931 - accuracy: 0.3302 - val_loss: 4.0567 - val_accuracy: 0.0775
Epoch 30/40
225/225 [==============================] - 25s 112ms/step - loss: 2.6136 - accuracy: 0.3581 - val_loss: 4.1480 - val_accuracy: 0.0900
Epoch 31/40
225/225 [==============================] - 25s 111ms/step - loss: 2.5639 - accuracy: 0.3637 - val_loss: 4.0887 - val_accuracy: 0.0775
Epoch 32/40
225/225 [==============================] - 25s 112ms/step - loss: 2.5046 - accuracy: 0.3867 - val_loss: 4.1899 - val_accuracy: 0.0887
Epoch 33/40
225/225 [==============================] - 25s 112ms/step - loss: 2.5211 - accuracy: 0.3845 - val_loss: 4.2925 - val_accuracy: 0.0825
Epoch 34/40
225/225 [==============================] - 26s 115ms/step - loss: 2.4888 - accuracy: 0.3946 - val_loss: 4.3284 - val_accuracy: 0.0875
Epoch 35/40
225/225 [==============================] - 31s 138ms/step - loss: 2.4185 - accuracy: 0.4089 - val_loss: 4.2538 - val_accuracy: 0.0950
Epoch 36/40
225/225 [==============================] - 30s 131ms/step - loss: 2.4273 - accuracy: 0.4033 - val_loss: 4.3752 - val_accuracy: 0.0900
Epoch 37/40
225/225 [==============================] - 29s 131ms/step - loss: 2.4518 - accuracy: 0.3986 - val_loss: 4.4235 - val_accuracy: 0.0925
Epoch 38/40
225/225 [==============================] - 29s 129ms/step - loss: 2.3968 - accuracy: 0.4104 - val_loss: 4.3973 - val_accuracy: 0.0838
Epoch 39/40
225/225 [==============================] - 29s 130ms/step - loss: 2.3868 - accuracy: 0.4178 - val_loss: 4.6153 - val_accuracy: 0.0900
Epoch 40/40
225/225 [==============================] - 29s 128ms/step - loss: 2.3420 - accuracy: 0.4216 - val_loss: 4.5528 - val_accuracy: 0.0825
"""
