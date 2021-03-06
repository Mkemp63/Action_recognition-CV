import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

import config
import test


def countLayers(model, type):
    ans = 0
    for layer in model.layers:
        if isinstance(layer, type): ans += 1
    return ans


def makeSubModel(model, after_layer_type, after_layer_number, freeze: bool):
    aantal = 0
    if after_layer_number < 0: after_layer_number = countLayers(model, after_layer_type) + 1 + after_layer_number
    m_new = models.Sequential()
    for layer in model.layers:
        m_new.add(layer)
        if isinstance(layer, after_layer_type):
            aantal += 1
        if aantal == after_layer_number:
            break
    if freeze:
        for layer in m_new.layers:
            layer.trainable = False
    print(aantal)
    return m_new


def checkValues(new_layers_a_list, new_layers_b_list, aantal_fusion, type_fusion_list):
    ans = ""
    if aantal_fusion != len(type_fusion_list):
        ans = f"Aantal fusion moet gelijk zijn aan len(type_fusion_list); {aantal_fusion} & {len(type_fusion_list)}"
    elif aantal_fusion != len(new_layers_a_list):
        ans = f"Aantal fusion moet gelijk zijn aan len(new_layers_a_list); {aantal_fusion} & {len(new_layers_a_list)}"
    elif len(new_layers_b_list) != len(new_layers_a_list):
        ans = f"a_list must have same size as b_list.\
         len a & b: {len(new_layers_a_list)} & {len(new_layers_a_list)}"

    return ans


def makeFusionModel(model_a, model_b, after_layer_type_list, aft_lay_n_a=-1, aft_lay_n_b=-1, new_layers_a_list=[],
                    new_layers_b_list=[], freeze: bool = False, aantal_fusion: int = 1,
                    type_fusion_list: list = ['conc'],
                    output_layer=layers.Dense(4, activation='softmax', name="Dense_output4")):
    """ WARNING: ZORG ERVOOR DAT WAAR JE DE MODELLEN AFSNIJDT ZE WEL DAADWERKELIJK KUNNEN FUSEREN
    model_a / model_b = twee modellen om samen te voegen
    after_layer_type = type laag waar het achter moet komen, bijv: conv.., max..., dense..
    after_layer_number = na hoeveelste van type de fusion moet komen
        -1 => na laatste van dat type
    new_layers = lijst met layers
    type_fusion: 'conc_conv', 'conc_dense'(concatenate), 'avg', 'add', 'sub'(tract), 'max', 'min'
    """
    # Checken of new layers niet leeg is
    correctNess = checkValues(new_layers_a_list, new_layers_b_list, aantal_fusion, type_fusion_list)
    if correctNess == "":
        part_a = makeSubModel(model_a, after_layer_type_list[0], aft_lay_n_a, freeze)
        part_b = makeSubModel(model_b, after_layer_type_list[1], aft_lay_n_b, freeze)

        a = part_a.output
        b = part_b.output
        for i in range(0, aantal_fusion):
            fusion_now = type_fusion_list[i]

            print(f"Layer... {fusion_now}")
            print(f"shapes: {a.shape} & {b.shape} len: {len(a.shape)} & {len(b.shape)}")
            # CHECK
            if (fusion_now == 'conc_conv' and (len(a.shape) < 4 or len(b.shape) < 4)) or \
                    (fusion_now == 'conc_dense' and (len(a.shape) > 2 or len(b.shape) > 2)):
                print("ERROR: USING WRONG CONCATENTATE")

            if fusion_now == 'conc_conv':
                merge = layers.concatenate(axis=3, inputs=[a, b])
            elif fusion_now == 'conc_dense':
                merge = layers.concatenate(axis=1, inputs=[a, b])
            elif fusion_now == 'avg':
                if a.shape != b.shape:
                    b = b if b.shape[1] < a.shape[1] else layers.Dense(a.shape[1], activation="relu")(b)
                    a = a if a.shape[1] < b.shape[1] else layers.Dense(b.shape[1], activation='relu')(a)
                merge = layers.average(inputs=[a, b])
            elif fusion_now == 'add':
                if a.shape != b.shape:
                    b = b if b.shape[1] < a.shape[1] else layers.Dense(a.shape[1], activation="relu")(b)
                    a = a if a.shape[1] < b.shape[1] else layers.Dense(b.shape[1], activation='relu')(a)
                merge = layers.add(inputs=[a, b])
            elif fusion_now == 'sub':
                if a.shape != b.shape:
                    b = b if b.shape[1] < a.shape[1] else layers.Dense(a.shape[1], activation="relu")(b)
                    a = a if a.shape[1] < b.shape[1] else layers.Dense(b.shape[1], activation='relu')(a)
                merge = layers.subtract(inputs=[a, b])
            elif fusion_now == 'max':
                if a.shape != b.shape:
                    b = b if b.shape[1] < a.shape[1] else layers.Dense(a.shape[1], activation="relu")(b)
                    a = a if a.shape[1] < b.shape[1] else layers.Dense(b.shape[1], activation='relu')(a)
                merge = layers.maximum(inputs=[a, b])
            elif fusion_now == 'min':
                if a.shape != b.shape:
                    b = b if b.shape[1] < a.shape[1] else layers.Dense(a.shape[1], activation="relu")(b)
                    a = a if a.shape[1] < b.shape[1] else layers.Dense(b.shape[1], activation='relu')(a)
                merge = layers.minimum(inputs=[a, b])
            a = merge
            print(f"merge: {merge.shape}")
            for j in range(0, len(new_layers_a_list[i])):
                a = new_layers_a_list[i][j](a)

            for j in range(0, len(new_layers_b_list[i])):
                b = new_layers_b_list[i][j](b)
        print(a.shape)
        x = layers.Dense(4, activation='softmax', name='Dense_outputx')(a)
        model_new = models.Model(inputs=[part_a.input, part_b.input], outputs=x)
        model_new.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                          metrics=['accuracy'])
        return model_new
    else:
        print(correctNess)
        return


def standardModel4(modela, modelb, printen: bool, kernel_reg, fusions: list = []):
    cut_off_layers = [layers.Dropout, layers.Flatten]
    after_layer_N_a = 1  # i = na i van cut_off_layer[0]; -2 betekent: voor de laatste van cut_off_layer[0]
    after_layer_N_b = 1  # i = na i van cut_off_layer[0]; -2 betekent: voor de laatste van cut_off_layer[1]

    aantal_fusion = 1
    fusion_types = ['conc_dense'] if len(fusions) < 1 else [fusions[0]]

    # exclusief output layer, die zit al in de functie
    path_a = [[layers.Dense(40, activation='relu', kernel_regularizer=kernel_reg, name='Dense1')]]
    path_b = [[]]
    new_model = makeFusionModel(modela, modelb, cut_off_layers, after_layer_N_a, after_layer_N_b,
                                path_a, path_b,
                                False, aantal_fusion, fusion_types)
    if printen:
        print(new_model.summary())
    return new_model


def alt_model(modela, modelb, printen: bool, kernel_reg: bool, fusions: list = []):
    """Gebaseerd op rechter model assignment 5 Q&A"""
    cut_off_layers = [layers.Conv2D, layers.Conv2D]
    after_layer_N_a = 3  # i = na i van cut_off_layer[0];
    after_layer_N_b = 5  # i = na i van cut_off_layer[0];

    aantal_fusion = 2
    # fusion_types = ['conc_conv', 'conc_dense']  # concatenate, kan je aanpassen met iets anders
    fusion_types = ['conc_conv', 'conc_dense'] if len(fusions) < 2 else [fusions[0], fusions[1]]

    # exclusief output layer, die zit al in de functie
    path_a = [[layers.Conv2D(kernel_size=(3, 3), filters=2, activation='relu', kernel_regularizer=kernel_reg),
               layers.MaxPooling2D((2, 2)),
               layers.Conv2D(2, (3, 3), activation='relu', kernel_regularizer=kernel_reg),
               layers.Flatten(),
               layers.Dense(60, activation='relu', kernel_regularizer=kernel_reg)],
              []]
    path_b = [[layers.Conv2D(kernel_size=(3, 3), filters=2, activation='relu', kernel_regularizer=kernel_reg),
               layers.MaxPooling2D((2, 2)),
               layers.Flatten(),
               layers.Dense(30, activation='relu', kernel_regularizer=kernel_reg)],
              []]
    new_model = makeFusionModel(modela, modelb, cut_off_layers, after_layer_N_a, after_layer_N_b,
                                path_a, path_b,
                                False, aantal_fusion, fusion_types)
    if printen:
        print(new_model.summary())
    return new_model


def flatten_model(modela, modelb, printen: bool, kernel_reg, fusions: list = []):
    cut_off_layers = [layers.Flatten, layers.Flatten]
    after_layer_N_a = 1  # i = na i van cut_off_layer[0]; -2 betekent: voor de laatste van cut_off_layer[0]
    after_layer_N_b = 1  # i = na i van cut_off_layer[0]; -2 betekent: voor de laatste van cut_off_layer[1]

    aantal_fusion = 1
    fusion_types = ['conc_dense'] if len(fusions) < 1 else [fusions[0]]

    # exclusief output layer, die zit al in de functie
    path_a = [[layers.Dense(100, activation='relu', kernel_regularizer=kernel_reg),
               layers.Dense(40, activation='relu', kernel_regularizer=kernel_reg)]]
    path_b = [[]]
    new_model = makeFusionModel(modela, modelb, cut_off_layers, after_layer_N_a, after_layer_N_b,
                                path_a, path_b,
                                False, aantal_fusion, fusion_types)
    if printen:
        print(new_model.summary())
    return new_model


def probeerFusionOpties(model2, model3, printen: bool, train_frame, train_flow, train_l, train_flow_l, kernel_reg=None):
    opties = ['conc_dense', 'avg', 'add', 'sub', 'max', 'min']
    tr, val, tr_l, val_l = train_test_split(train_frame, train_l, test_size=config.Validate_perc, random_state=42)
    print(train_frame.shape)
    print(train_flow.shape)
    print(train_flow_l.shape)
    print(train_l.shape)
    tr_flow, val_flow, tr_l_flow, val_l_flow = train_test_split(train_flow, train_flow_l,
                                                                test_size=config.Validate_perc, random_state=42)
    for i in range(0, 2):
        for j in range(0, len(opties)):
            model4 = standardModel4(model2, model3, True, kernel_reg, fusions=[opties[j]]) if i == 0 else \
                flatten_model(model2, model3, True, kernel_reg, fusions=[opties[j]])

            hist_model4 = model4.fit([tr, tr_flow], tr_l, validation_data=([val, val_flow], val_l),
                                     epochs=6)
            test_loss_m4, test_acc_m4 = model4.evaluate([val, val_flow], val_l)
            print(f"Test acc: {test_acc_m4} & loss: {test_loss_m4}")
    print("Test alt model ......")
    opties2 = [['conc_conv', 'conc_dense'], ['avg', 'avg'], ['add', 'add'], ['sub', 'sub'], ['max', 'max'],
               ['min', 'min']]
    for j in range(0, len(opties2)):
        model4 = alt_model(model3, model2, True, kernel_reg, fusions=opties2[j])
        hist_model4 = model4.fit([tr_flow, tr], tr_l, validation_data=([val_flow, val], val_l), epochs=6)
        test_loss_m4, test_acc_m4 = model4.evaluate([val_flow, val], val_l)
        print(f"Test acc: {test_acc_m4} & loss: {test_loss_m4}")
    print("Done testing")


def printNiks():
    print()


def testFusion():
    modela = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)
    modelb = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)

    new_model = makeFusionModel(modela, modelb, [layers.MaxPooling2D, layers.MaxPooling2D], 3, 3,
                                [[layers.Conv2D(kernel_size=(3, 3), filters=8, activation='relu'),
                                  layers.MaxPooling2D((2, 2)),
                                  layers.Conv2D(kernel_size=(3, 3), filters=8, activation='relu'),
                                  layers.Flatten()], [layers.Dense(60, activation='relu')]
                                 ], [[layers.Conv2D(kernel_size=(3, 3), filters=16, activation='relu'),
                                      layers.MaxPooling2D((2, 2)),
                                      layers.Conv2D(kernel_size=(3, 3), filters=16, activation='relu'),
                                      layers.Flatten()], []],
                                False, 2, ['conc_conv', 'conc_dense'])
    print(new_model.summary())
    input()
    return


def testDingIets():
    modela = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)
    modelb = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)
    alt_model(modela, modelb, True, None)  # standardModel4(modela, modelb, True)
    input()

    testFusion()

    model = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)
    print(model.summary())
    input()
    m_new = makeSubModel(model, layers.Conv2D, -1, True)
    print(m_new.summary())

