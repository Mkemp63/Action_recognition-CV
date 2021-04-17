import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from tensorflow.keras import layers, models

import HelperFunctions as HF
import OpticalFlow as OptF
import fusion
import config
import data
import augmentationMethods as am

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


def make_baseline_model(input_shape, hidden_layer_neurons, activation1='relu', activation2='relu', activation3='relu',
                        optimizer=tf.keras.optimizers.Adam(), conv_layers=2, filter_size=16, kernel_size=3, dropout=0.0,
                        output_size=40,
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
    model.add(layers.Dense(hidden_layer_neurons, activation=activation2, kernel_regularizer=k_reg))
    model.add(layers.Dense(hidden_layer_neurons, activation=activation2, kernel_regularizer=k_reg))
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
    newImgs, labss = data.getDataSet(["applauding_004.jpg"], config.STANF_CONV_CROP, False, [1])
    applauding = cv2.imread(config.STANF_CONV_CROP + "applauding_004.jpg", 1)
    #img_blur = blur(image=applauding)
    img_hue = am.adjustHue(img=applauding, hue=-.5)
    # img = seq(images=np.array([applauding]))
    # print(type(img))
    # print(img.shape)
    cv2.imshow("img", img_hue)
    cv2.waitKey(0)
    # for i in img:
    #     print(type(i))
    #     # print(i)
    #     cv2.imshow("img", i)
    #     cv2.waitKey(0)
    print("DONE")
    cv2.destroyAllWindows()


def plot_val_train_loss(history, title, save: bool = False):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')
    plt.title(title)
    if save:
        plt.savefig(f"./Plots/{title}.png")
    plt.show()


def plot_val_train_acc(history, title, save: bool = False):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.title(title)
    if save:
        plt.savefig(f"./Plots/{title}.png")
    plt.show()


def hardNegativeMining(model, train, labels):
    negatives = []
    neg_labels = []
    print(f"Length labels: {labels.shape[0]}")
    for i in range(0, labels.shape[0]):
        pred = np.argmax(model.predict(np.array([train[i]])), axis=1)
        if i < 5:
            print(f"Pred: {pred} vs labels: {labels[i]}")
        if pred != labels[i]:
            negatives.append(train[i])
            neg_labels.append(labels[i])
    return np.array(negatives), np.array(neg_labels)


def testKernelReg(stanf_train, stanf_train_lab, stanf_test, stanf_test_lab, tv_train_imgs, tv_train_labels,
                    tv_val_imgs, tv_val_labels, input_shape):
    opties = [-1, 0.001, 0.01, 0.1]
    for i in range(0, 4):
        kernel_regulariser = tf.keras.regularizers.l2(opties[i]) if i > 0 else None
        print(f"Testen met kernel regulariser value: {opties[i]}  <<-- (-1 = None)")
        makeModelsFinal(stanf_train, stanf_train_lab, stanf_test, stanf_test_lab, tv_train_imgs, tv_train_labels,
                    tv_val_imgs, tv_val_labels, input_shape, kernel_reg=kernel_regulariser)
    print("Testing Kernel regularisation is done!")
    return


def makeModelsFinal(tv_train_imgs, tv_train_labels, tv_val_imgs, tv_val_labels, input_shape, kernel_reg, leave_one_out: bool = False, test_fusions: bool = False):
    # Make first model
    print("Making the first model...")
    # Variabelen moeten wel aangepast worden waarschijnlijk
    clr = cyclical_learning_rate(1e-4, 1e-3, step_size=4, scale_fn=lambda x: 1, scale_mode='cycle', type="triangle2")
    model_base = make_baseline_model(input_shape, kernel_size=3, optimizer=tf.keras.optimizers.Adam(learning_rate=clr),
                                     hidden_layer_neurons=60, activation3='softmax', conv_layers=3, filter_size=8,
                                     dropout=0.5, dubbel_conv=True, filter_multiplier=2)

    if leave_one_out:
        for i in range(1, 12):
            if i == 1 or i == 2 or i == 10 or i == 11:
                print(i)
                model_current = models.clone_model(model_base)
                model_current.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=clr),
                                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                      metrics=['accuracy'])

                stanf_train, stanf_train_lab, stanf_test, stanf_test_lab = data.getStanfordData(leave_one_out=True,
                                                                                                leave_out_ind=i)
                stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stanf_train,
                                                                                                  stanf_train_lab,
                                                                                                  test_size=config.Validate_perc)

                hist_base = model_current.fit(stanf_train, stanf_train_lab,
                                                       validation_data=(stanf_test, stanf_test_lab),
                                                       epochs=config.Epochs, batch_size=config.Batch_size)

                test_loss_base, test_acc_base = model_current.evaluate(stanf_test, stanf_test_lab)
                print(f"Test acc: {test_acc_base} & loss: {test_loss_base} for leave_out_ind {i}")

                # Plot test loss and accuracy
                plot_val_train_loss(hist_base, f"Base model {i} training and validation loss", save=True)
                plot_val_train_acc(hist_base, f"Base model {i} training and validation accuracy", save=True)

    else:
        stanf_train, stanf_train_lab, stanf_test, stanf_test_lab = data.getStanfordData()
        stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stanf_train, stanf_train_lab,
                                                                                          test_size=config.Validate_perc)
        hist_base, model_base = model_base.fit(stf_train_imgs, stf_train_labels,
                                               validation_data=(stf_val_imgs, stf_val_labels),
                                               epochs=config.Epochs, batch_size=config.Batch_size)
        test_loss_base, test_acc_base = model_base.evaluate(stanf_test, stanf_test_lab)
        print(f"Test acc: {test_acc_base} & loss: {test_loss_base}")

        # Plot test loss and accuracy
        plot_val_train_loss(hist_base, "Base model training and validation loss", save=True)
        plot_val_train_acc(hist_base, "Base model training and validation accuracy", save=True)

    # Make second model, based on transfer learning
    print("Making the second model...")
    tv_output_layer = layers.Dense(4, activation="softmax", name="Dense_output")
    model_tl = transfer_learn_model(config.MODELS + "Baseline.h5", tv_output_layer, freeze=True)
    hist_tl, model_tl = model_tl.fit(tv_train_imgs, tv_train_labels, epochs=config.Epochs,
                                         validation_data=(tv_val_imgs, tv_val_labels), batch_size=config.Batch_size)
    model_tl.save(config.MODELS + "TV_TL.h5")
    plot_val_train_loss(hist_tl, title="Transfer learn model training and validation loss", save=True)
    plot_val_train_acc(hist_tl, title="Transfer model training and validation accuracy", save=True)


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
    plot_val_train_loss(hist_model3, "Model 3 training and validation loss", save=True)
    plot_val_train_acc(hist_model3, "Model 3 training and validation accuracy", save=True)



    """Make fourth model, etc.."""
    # ALS WE EEN DROPOUT LAYER GEBRUIKEN IN MODEL 1/2/3 DAN MOET FUSION.STANDARDMODEL MISSCHIEN AANGEPAST WORDEN
    # WANT DAAR HOUDT HIJ NU GEEN REKENING MEE.

    print("Making the fourth model...")
    # model4 = fusion.standardModel4(model2, model3, True, kernel_reg)

    # Alternatief op basis van Assignment 5 Q&A
    # model4 = fusion.standardModel4(model3, model2, True, kernel_reg)

    # train_data_model4 = [data_2, data_3]
    # labels_model4 = .... zou hetzelfde moeten zijn als voor model2 en model3
    # hist_model4, model4 = model4.fit(train_data_model4, labels_model4, epochs=config.Epochs)
    # test_loss_m4, test_acc_m4 = model4.evaluate(stanf_test, stanf_test_lab)
    # print(f"Test acc: {test_acc_m4} & loss: {test_loss_m4}")

    # Plot test loss and accuracy
    # plot_val_train_loss(hist_model4, "Complete model training and validation loss", save=True)
    # plot_val_train_acc(hist_model4, "Complete model training and validation accuracy", save=True)

    # Testen fusion
    if test_fusions:
        fusion.probeerFusionOpties(model2, model3, True, train_data_model4, labels_model4)

    print()
    return


def main():
    input_shape = (config.Image_size, config.Image_size, 3)
    # Test Optical Flow
    # testOpticalFlow()

    # stf_trainset: whole training set (incl. val); stf_train_imgs: only training (excl val)
    # print("Get DataSet")
    # stf_trainset, stf_trainset_lab, stf_test, stf_test_lab = data.getStanfordData()
    # stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stf_trainset, stf_trainset_lab,
    #                                                                                   test_size=config.Validate_perc)

    """ 
    ALS JE EEN KEER KERNEL REGULARISATION & FUSION MODELS WILT TESTEN
    """
    testKernelReg()
    # Function to train and make the four models
    # makeModelsFinal(stf_trainset, stf_trainset_lab, stf_test, stf_test_lab, input_shape)

    # Test the different forms of augmentation
    #test()

    # Choice task cyclical learning rate scheduler
    # clr = cyclical_learning_rate(1e-4, 1e-3, step_size=10, scale_fn=lambda x: 1, scale_mode='cycle', type="cyclical")
    # print("start random search")
    # random_search(input_shape, stf_trainset, stf_trainset_lab, stf_test, stf_test_lab, n_iter=10)

    # print("Make model")
    # model = make_baseline_model(input_shape, conv_layers=3, hidden_layer_neurons=60, activation3='softmax',
    #                             k_reg=tf.keras.regularizers.l2(0.01))
    # model_result = model.fit(stf_train_imgs, stf_train_labels, epochs=config.Epochs,
    #                          validation_data=(stf_val_imgs, stf_val_labels), batch_size=config.Batch_size)

    # if not os.path.isfile(config.MODELS + "Baseline.h5"):
    #     print("Model file not found, creating...")
    #     model = make_baseline_model(input_shape, kernel_size=3, optimizer='adam', hidden_layer_neurons=60,
    #                                 activation3='softmax', conv_layers=3, filter_size=8, dropout=0.5,
    #                                 dubbel_conv=True, filter_multiplier=2)
    #
    #     model_result = model.fit(stf_trainset, stf_trainset_lab, validation_data=(stf_test, stf_test_lab),
    #                              epochs=config.Epochs, batch_size=config.Batch_size)
    #     model.save(config.MODELS + "Baseline.h5")
    # else:
    #     print("Model file located")
    #     model = models.load_model(config.MODELS + "Baseline.h5")

    # y_pred = model.predict(stf_test)
    # y_pred = np.argmax(y_pred, axis=1)
    # confusion_matrix = sklearn.metrics.confusion_matrix(stf_test_lab, y_pred)
    # f = open("cf.txt", 'a')
    # p = ""
    # for row in confusion_matrix:
    #     for cell in row:
    #         p += str(cell) + "; "
    #     p += "\n"
    # f.write(p)
    # f.close()
    # input()
    # test_loss, test_acc = test_model(model, stf_test, stf_test_lab)

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

    newImgs, labss = data.getDataSet(tv_train_files, config.TV_CONV_CROP, False, tv_train_labels_ind)
    tv_train_imgs, tv_val_imgs, tv_train_labels, tv_val_labels = train_test_split(newImgs, labss,
                                                                                  test_size=config.Validate_perc)

    # Function to train and make the four models
    makeModelsFinal(tv_train_imgs, tv_val_imgs, tv_train_labels, tv_val_labels, input_shape, kernel_reg=1,
                    leave_one_out=True)

    # if not os.path.isfile(config.MODELS + "TV_TL.h5"):
    #     print("Model file not found, creating...")
    #     tv_output_layer = layers.Dense(4, activation="softmax", name="Dense_output")
    #     tl_model = transfer_learn_model(config.MODELS + "Baseline.h5", tv_output_layer, freeze=True, lr=clr)
    #     tl_model_result = tl_model.fit(tv_train_imgs, tv_train_labels, epochs=config.Epochs,
    #                                    validation_data=(tv_val_imgs, tv_val_labels), batch_size=config.Batch_size)
    #     tl_model.save(config.MODELS + "TV_TL.h5")
    # else:
    #     print("Model file located")
    #     tl_model = models.load_model(config.MODELS + "TV_TL.h5")
    #
    # print("Do you want to start grid search? Press any key")
    # input()

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

