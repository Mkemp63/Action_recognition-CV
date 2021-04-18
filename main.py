import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import tensorflow_addons as tfa
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import Callback

import HelperFunctions as HF
import OpticalFlow as OptF
import augmentationMethods as am
import config
import data
import fusion


def transfer_learn_model(model_path, new_output_layer, freeze: bool = False, lr=0.001):
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
                        output_size=40, filter_multiplier=1, dubbel_conv: bool = True, k_reg=None, dense_n: int = 2):
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
        model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1,
                                kernel_regularizer=k_reg))
        model.add(layers.MaxPooling2D((2, 2)))
        filter_size = int(filter_size * filter_multiplier)
    model.add(layers.Conv2D(filter_size, (kernel_size, kernel_size), activation=activation1, kernel_regularizer=k_reg))
    model.add(layers.Flatten())
    for i in range(dense_n):
        model.add(layers.Dense(hidden_layer_neurons, activation=activation2, kernel_regularizer=k_reg))
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_size, activation=activation3))
    model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    print(model.summary())
    return model


def make_small_model():
    stanf_train, stanf_train_lab, stanf_test, stanf_test_lab = data.getStanfordData()

    HF.take_middle_frame(config.TV_VIDEOS)
    tv_test_files, tv_test_labels, tv_train_files, tv_train_labels = data.train_tests_tv()

    tv_test_files, tv_train_files = video_name_to_image_name(tv_test_files, tv_train_files)

    tv_test_labels_ind = HF.convertLabel(tv_test_labels)
    tv_train_labels_ind = HF.convertLabel(tv_train_labels)

    tv_train, tv_train_l = data.getDataSet(tv_train_files, config.TV_CONV_CROP, False, tv_train_labels_ind)
    tv_test, tv_test_l = data.getDataSet(tv_test_files, config.TV_CONV_CROP, False, tv_test_labels_ind, False)

    model_base = make_baseline_model((112, 112, 3), kernel_size=3,
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     hidden_layer_neurons=60, activation3='softmax', conv_layers=3, filter_size=8,
                                     dropout=0.5, dubbel_conv=True, filter_multiplier=1.5, k_reg=None, dense_n=1)

    hist_base = model_base.fit(stanf_train, stanf_train_lab,
                               validation_data=(stanf_test, stanf_test_lab),
                               epochs=config.Epochs, batch_size=config.Batch_size)
    model_base.save(config.MODELS + "Small.h5")
    test_loss_base, test_acc_base = model_base.evaluate(stanf_test, stanf_test_lab)
    print(f"Test acc: {test_acc_base} & loss: {test_loss_base}")

    # Plot test loss and accuracy
    plot_val_train_loss(hist_base, "Base model training and validation loss", save=True)
    plot_val_train_acc(hist_base, "Base model training and validation accuracy", save=True)

    tv_output_layer = layers.Dense(4, activation="softmax", name="Dense_output")
    model_tl = transfer_learn_model(config.MODELS + "Small.h5", tv_output_layer, freeze=True)
    hist_tl = model_tl.fit(tv_train, tv_train_l, epochs=config.Epochs,
                           validation_data=(tv_test, tv_test_l), batch_size=config.Batch_size)


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
                                                             scale_mode=scale_mode)
    elif type == "triangle":
        clr = tfa.optimizers.TriangularCyclicalLearningRate(initial_learning_rate=initial_learning_rate,
                                                            maximal_learning_rate=maximal_learning_rate,
                                                            step_size=step_size,
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


def video_name_to_image_name(testx, trainx):
    # very unoptimized, but it works :)
    for i in range(len(testx)):
        name = testx[i]
        name = name[:-4] + ".jpg"
        testx[i] = name

    for i in range(len(trainx)):
        name = trainx[i]
        name = name[:-4] + ".jpg"
        trainx[i] = name

    return testx, trainx


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
    img_hue = am.adjustHue(img=applauding, hue=-.5)

    cv2.imshow("img", img_hue)
    cv2.waitKey(0)
    print("DONE")
    cv2.destroyAllWindows()


class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc or val_acc reaches a specified baseline
    """

    def __init__(self, monitor='accuracy', monitor2="accuracy", baseline=0.9, baseline2=0.9):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.monitor2 = monitor2
        self.baseline = baseline
        self.baseline2 = baseline2

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = logs.get(self.monitor)
        acc2 = logs.get(self.monitor2)
        if acc is not None and acc2 is not None:
            if acc >= self.baseline and acc2 >= self.baseline2:
                print('Epoch %d: Reached baseline, terminating training' % (epoch))
                self.model.stop_training = True


def make_flow_model(k_reg=None):
    model = models.Sequential()
    model.add(layers.MaxPooling2D(input_shape=(112, 112, 20), pool_size=(2, 2), strides=(2, 2), name='pool1'))  # 56
    model.add(layers.Conv2D(6, (5, 5), activation='relu', kernel_regularizer=k_reg, name='conv1'))  # 52
    model.add(layers.MaxPooling2D((2, 2), name='pool2'))  # 26
    model.add(layers.Conv2D(12, (3, 3), activation='relu', kernel_regularizer=k_reg, name='conv2'))  # 24
    model.add(layers.MaxPooling2D((2, 2), name='pool3'))  # 12
    model.add(layers.Conv2D(20, (3, 3), activation='relu', kernel_regularizer=k_reg, name='conv3'))  # 10
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=k_reg, name='conv4'))  # 8
    model.add(layers.MaxPooling2D((2, 2), name='pool4'))  # 4
    model.add(layers.Flatten(name="plat"))
    model.add(layers.Dense(4, activation='softmax', name="denz1"))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    print(model.summary())
    return model


def loop_model3(flow_train, flow_train_l, flow_test, flow_test_l, k_reg):
    while True:
        model = make_flow_model(k_reg=k_reg)
        model.fit(flow_train, flow_train_l, epochs=5, validation_data=(flow_test, flow_test_l),
                  batch_size=config.Batch_size)
        loss, acc = model.evaluate(flow_test, flow_test_l)
        if acc > 0.32:
            return model


def testenModel3():
    flow_train, flow_train_l, flow_test, flow_test_l = data.loadFlowData(False, True, threeD=False)

    callbacks = [TerminateOnBaseline(monitor='val_accuracy', baseline=0.33, monitor2="accuracy", baseline2=0.33)]

    model = make_flow_model()
    model.save(config.MODELS + "EarlyStop3.h5")

    model.fit(flow_train, flow_train_l, epochs=config.Epochs, validation_data=(flow_test, flow_test_l),
              batch_size=config.Batch_size, callbacks=callbacks)
    y_pred = model.predict(flow_test)
    y_pred = np.argmax(y_pred, axis=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(flow_test_l, y_pred)
    print(y_pred)
    print(confusion_matrix)
    tr_pred = model.predict(flow_train)
    tr_pred = np.argmax(tr_pred, axis=1)
    confusion_matrix = sklearn.metrics.confusion_matrix(flow_train_l, tr_pred)
    print(y_pred)
    print(confusion_matrix)
    n_train, n_labels = hardNegativeMining2(model, flow_train, flow_train_l)
    print(n_labels)
    model.fit(n_train, n_labels, epochs=10, validation_data=(flow_test, flow_test_l),
              batch_size=config.Batch_size, shuffle=True)

    print("Read flow data")
    input()


def hardNegativeMining2(model, train, labels):
    negatives = []
    neg_labels = []
    print(f"Length labels: {labels.shape[0]}")
    preds = np.argmax(model.predict(train), axis=1)
    for i in range(0, labels.shape[0]):
        if preds[i] != labels[i]:
            negatives.append(train[i])
            neg_labels.append(labels[i])
    return np.array(negatives), np.array(neg_labels)


def plot_val_train_loss(history, title, save: bool = False):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower left')
    plt.tight_layout()
    # plt.title(title)
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
    plt.tight_layout()
    # plt.title(title)
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


def testKernelReg(tv_train_imgs, tv_train_labels, tv_test_imgs, tv_test_labels, input_shape,
                  flow_train, flow_train_l, flow_test, flow_test_l, aantal_frames):
    opties = [-1, 0.001, 0.01, 0.1]
    for i in range(0, 4):
        kernel_regulariser = tf.keras.regularizers.l2(opties[i]) if i > 0 else None
        print(f"Testen met kernel regulariser value: {opties[i]}  <<-- (-1 = None)")
        makeModelsFinal(tv_train_imgs, tv_train_labels, tv_test_imgs, tv_test_labels, input_shape,
                        flow_train, flow_train_l, flow_test, flow_test_l, aantal_frames, kernel_reg=kernel_regulariser,
                        useVal=True)
    print("Testing Kernel regularisation is done!")
    return


def makeModelsFinal(tv_train_imgs, tv_train_labels, tv_test_imgs, tv_test_labels, input_shape,
                    flow_train, flow_train_l, flow_test, flow_test_l, aantal_frames, kernel_reg=None,
                    leave_one_out: bool = False, test_fusions: bool = False, useVal: bool = False):
    stanf_train, stanf_train_lab, stanf_test, stanf_test_lab = data.getStanfordData()

    if useVal:
        stf_train_imgs, stf_val_imgs, stf_train_labels, stf_val_labels = train_test_split(stanf_train, stanf_train_lab,
                                                                                          test_size=config.Validate_perc,
                                                                                          random_state=42)
        tv_train_imgs, tv_val_imgs, tv_train_labels, tv_val_labels = train_test_split(tv_train_imgs, tv_train_labels,
                                                                                      test_size=config.Validate_perc,
                                                                                      random_state=42)
        flow_train_vids, flow_val_vids, flow_train_labels, flow_val_labels = train_test_split(flow_train, flow_train_l,
                                                                                              test_size=config.Validate_perc,
                                                                                              random_state=42)

    # Make first model
    print("Making the first model...")
    # Variabelen moeten wel aangepast worden waarschijnlijk
    clr = cyclical_learning_rate(1e-4, 1e-3, step_size=4, scale_fn=lambda x: 1, scale_mode='cycle', type="triangle2")
    model_base = make_baseline_model((112, 112, 3), kernel_size=3,
                                     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                     hidden_layer_neurons=60, activation3='softmax', conv_layers=3, filter_size=8,
                                     dropout=0.5, dubbel_conv=True, filter_multiplier=1.5, k_reg=None, dense_n=1)

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
        if useVal:
            hist_base = model_base.fit(stf_train_imgs, stf_train_labels,
                                       validation_data=(stf_val_imgs, stf_val_labels),
                                       epochs=config.Epochs, batch_size=config.Batch_size)

        else:
            hist_base = model_base.fit(stanf_train, stanf_train_lab,
                                       validation_data=(stanf_test, stanf_test_lab),
                                       epochs=config.Epochs, batch_size=config.Batch_size)
            model_base.save(config.MODELS + "Baseline.h5")

        test_loss_base, test_acc_base = model_base.evaluate(stanf_test, stanf_test_lab)
        print(f"Test acc: {test_acc_base} & loss: {test_loss_base}")

        # Plot test loss and accuracy
        plot_val_train_loss(hist_base, "Base model training and validation loss", save=True)
        plot_val_train_acc(hist_base, "Base model training and validation accuracy", save=True)

    # Make second model, based on transfer learning
    print("Making the second model...")

    tv_output_layer = layers.Dense(4, activation="softmax", name="Dense_output")
    model_tl = transfer_learn_model(config.MODELS + "Baseline.h5", tv_output_layer, freeze=True)
    if useVal:
        hist_tl = model_tl.fit(tv_train_imgs, tv_train_labels, epochs=config.Epochs + 5,
                               validation_data=(tv_val_imgs, tv_val_labels), batch_size=config.Batch_size)
    else:
        hist_tl = model_tl.fit(tv_train_imgs, tv_train_labels, epochs=config.Epochs,
                               validation_data=(tv_test_imgs, tv_test_labels), batch_size=config.Batch_size)
    if not useVal:
        model_tl.save(config.MODELS + "TV_TL.h5")
    plot_val_train_loss(hist_tl, title="Transfer learn model training and validation loss", save=True)
    plot_val_train_acc(hist_tl, title="Transfer model training and validation accuracy", save=True)
    test_loss_tl, test_acc_tl = model_tl.evaluate(tv_test_imgs, tv_test_labels)
    print(f"Test acc: {test_acc_tl} & loss: {test_loss_tl}")
    model2 = model_tl

    # Make third model, etc..
    print("Making the third model...")
    """
    WAARDEN MOETEN GELIJK ZIJN AAN MODEL1, MAAR MAG NIET GETRAIND ZIJN VOLGENS MIJ.
    """
    tv_train, tv_train_l = flowDataTV()
    if useVal:
        flow_train_vids, flow_val_vids, flow_train_labels, flow_val_labels = train_test_split(flow_train, flow_train_l,
                                                                                              test_size=config.Validate_perc,
                                                                                              random_state=42)
        tv_train_x, tv_test_x, tv_train_y, tv_test_y = train_test_split(tv_train, tv_train_l,
                                                                        test_size=config.Validate_perc,
                                                                        random_state=42)

    callbacks = [TerminateOnBaseline(monitor='val_accuracy', baseline=0.33, monitor2="accuracy", baseline2=0.33)]
    model3 = make_flow_model(k_reg=kernel_reg)
    if useVal:
        hist_model3 = model3.fit(flow_train_vids, flow_train_labels, epochs=config.Epochs,
                                 validation_data=(flow_val_vids, flow_val_labels),
                                 batch_size=config.Batch_size, callbacks=callbacks)
    else:
        hist_model3 = model3.fit(flow_train, flow_train_l, epochs=config.Epochs,
                                 validation_data=(flow_test, flow_test_l),
                                 batch_size=config.Batch_size, callbacks=callbacks)
    if not useVal:
        model3.save(config.MODELS + "EarlyStop3.h5")

    test_loss_m3, test_acc_m3 = model3.evaluate(flow_test, flow_test_l)
    print(f"Test acc: {test_acc_m3} & loss: {test_loss_m3}")
    # Plot test loss and accuracy
    plot_val_train_loss(hist_model3, "Model 3 training and validation loss", save=True)
    plot_val_train_acc(hist_model3, "Model 3 training and validation accuracy", save=True)

    print("Making the fourth model...")
    model4 = fusion.standardModel4(model2, model3, True, kernel_reg)
    # Alternatief op basis van Assignment 5 Q&A
    labels_model4 = flow_train_labels if useVal else flow_train_l  # zou hetzelfde moeten zijn als voor model2 en model3

    if useVal:
        train_data_model4 = [tv_train_x, flow_train_vids]
        hist_model4 = model4.fit(train_data_model4, labels_model4, epochs=6,
                                 validation_data=([tv_test_x, flow_val_vids], flow_val_labels))
    else:
        train_data_model4 = [tv_train, flow_train]
        hist_model4 = model4.fit(train_data_model4, labels_model4, epochs=6,
                                 validation_data=([tv_test_imgs, flow_test], flow_test_l))
    if not useVal:
        model4.save(config.MODELS + "model4.h5")
    test_loss_m4, test_acc_m4 = model4.evaluate([tv_test_imgs, flow_test], flow_test_l)
    print(f"Test acc: {test_acc_m4} & loss: {test_loss_m4}")

    # Plot test loss and accuracy
    plot_val_train_loss(hist_model4, "Complete model training and validation loss", save=True)
    plot_val_train_acc(hist_model4, "Complete model training and validation accuracy", save=True)

    # Testen fusion
    if test_fusions and useVal:
        fusion.printNiks()  # Zodat hij niet weer de import fusion verwijdert
        fusion.probeerFusionOpties(model2, model3, True, tv_train, flow_train, flow_train_l, tv_train_l)

    print()
    return


def flowDataTV():
    tv_test_files, tv_test_labels, tv_train_files, tv_train_labels = data.train_tests_tv()
    tv_test_files, tv_train_files = video_name_to_image_name(tv_test_files, tv_train_files)
    tv_train_labels_ind = HF.convertLabel(tv_train_labels)
    tv_train, tv_train_l = data.getDataSet(tv_train_files, config.TV_CONV_CROP, False, tv_train_labels_ind, fuse=True,
                                           aug=False)
    return tv_train, tv_train_l


def printWeights(model, modelName: str):
    ans = ""
    i = 0
    for layer in model.layers:
        ans += str(model.layers[i].get_weights()) + "\n\n"
        print(model.layers[i].get_weights())
        print("~~~~~~~~~~~~~~~~~~~~~~")
        # print(model.layers[i].bias.numpy())
        # print("~~~~~~~~~~~~~~~~~~~~~~")
        # print(model.layers[i].bias_initializer)
        # print("~~~~~~~~~~~~~~~~~~~~~~")
        i += 1
    f = open(modelName + ".txt", "w")
    f.write(ans)
    f.close()


def main():
    input_shape = (config.Image_size, config.Image_size, 3)
    aantal_frames = 10
    flow_train, flow_train_l, flow_test, flow_test_l = data.loadFlowData(False, True, threeD=False)

    # Transfer learn to TV-HI data
    HF.take_middle_frame(config.TV_VIDEOS)
    tv_test_files, tv_test_labels, tv_train_files, tv_train_labels = data.train_tests_tv()

    tv_test_files, tv_train_files = video_name_to_image_name(tv_test_files, tv_train_files)

    tv_test_labels_ind = HF.convertLabel(tv_test_labels)
    tv_train_labels_ind = HF.convertLabel(tv_train_labels)

    tv_train, tv_train_l = data.getDataSet(tv_train_files, config.TV_CONV_CROP, False, tv_train_labels_ind)
    tv_test, tv_test_l = data.getDataSet(tv_test_files, config.TV_CONV_CROP, False, tv_test_labels_ind, False)

    allesTesten = False
    if allesTesten:
        print("Alles testen begint")

        print("Testing kernel regularisation")
        # testKernelReg(tv_train, tv_train_l, tv_test, tv_test_l, input_shape,
        #               flow_train, flow_train_l, flow_test, flow_test_l, aantal_frames)
        print("Testing different fusion layers")
        makeModelsFinal(tv_train, tv_train_l, tv_test, tv_test_l, input_shape,
                        flow_train, flow_train_l, flow_test, flow_test_l, aantal_frames, test_fusions=True, useVal=True)

    print_weights = True
    if print_weights:
        model1 = tf.keras.models.load_model(config.MODELS + "Baseline.h5")
        model2 = tf.keras.models.load_model(config.MODELS + "TV_TL.h5")
        model3 = tf.keras.models.load_model(config.MODELS + "EarlyStop3.h5")
        model4 = tf.keras.models.load_model(config.MODELS + "model4.h5")
        printWeights(model1, "Baseline")
        printWeights(model2, "Transfer Learn")
        printWeights(model3, "Optical flow")
        printWeights(model4, "Fusion")


if __name__ == '__main__':
    main()
