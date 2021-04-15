import tensorflow as tf
from tensorflow.keras import layers, models

import test


def countLayers(model, type):
    ans = 0
    for layer in model.layers:
        if isinstance(layer, type): ans += 1
    return ans


def makeSubModel(model, after_layer_type, after_layer_number, freeze: bool):
    aantal = 0
    if after_layer_number == -1: after_layer_number = countLayers(model, after_layer_type)
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


def makeFusionModel2(model_a, model_b, after_layer_type, after_layer_number=-1, new_layers1=[], new_layers2=[],
                    freeze: bool=False, type_fusion: str = 'conc'):
    """
    model_a / model_b = twee modellen om samen te voegen
    after_layer_type = type laag waar het achter moet komen, bijv: conv.., max..., dense..
    after_layer_number = na hoeveelste van type de fusion moet komen
        -1 => na laatste van dat type
    new_layers = lijst met layers
    type_fusion: 'con'(catenate), 'avg', 'max', 'min', 'add', 'sub'(tract)
    """


def checkValues(new_layers_a_list, new_layers_b_list, aantal_fusion, type_fusion_list):
    ans = ""
    if aantal_fusion != len(type_fusion_list):
        ans = f"Aantal fusion moet gelijk zijn aan len(type_fusion_list); {aantal_fusion} & {len(type_fusion_list)}"
    elif aantal_fusion != len(new_layers_a_list):
        ans = f"Aantal fusion moet gelijk zijn aan len(new_layers_a_list); {aantal_fusion} & {len(new_layers_a_list)}"
    elif len(new_layers_b_list) + 1 != len(new_layers_a_list):
        ans = f"a_list must have 1 more list because that last list is used after the final fusion layer.\
         len a & b: {len(new_layers_a_list)} & {len(new_layers_a_list)}"
    # for i in range(0, len(new_layers_a_list)):
    #     if len(new_layers_a_list[i]) == 0:
    #         ans = f""
    return ans


def makeFusionModel(model_a, model_b, after_layer_type_list, aft_lay_n_a=-1, aft_lay_n_b=-1, new_layers_a_list=[],
                    new_layers_b_list=[], freeze: bool = False, aantal_fusion: int = 1,
                    type_fusion_list: list = ['conc'], output_layer=layers.Dense(4, activation='softmax')):
    """ WARNING: ZORG ERVOOR DAT WAAR JE DE MODELLEN AFSNIJDT ZE WEL DAADWERKELIJK KUNNEN FUSEREN
    model_a / model_b = twee modellen om samen te voegen
    after_layer_type = type laag waar het achter moet komen, bijv: conv.., max..., dense..
    after_layer_number = na hoeveelste van type de fusion moet komen
        -1 => na laatste van dat type
    new_layers = lijst met layers
    type_fusion: 'conc_conv', 'conc_dense'(concatenate), 'avg', 'add', 'sub'(tract), 'max', 'min'
    """
    # Checken of new layers niet leeg is
    correctNess = checkValues(new_layers_a_list, new_layers_b_list,  aantal_fusion, type_fusion_list)
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
            # DEZE CHECK HIERONDER IS BELANGRIJK !
            elif a.shape != b.shape:
                print(f"To be able to use {fusion_now}, a and b their outputs needs to be the same size: {a.shape} & {b.shape}")
                print("Press key to end the method call and return nothing")
                input()
                return
            elif fusion_now == 'avg':
                merge = layers.average(inputs=[part_a.output, part_b.output])
            elif fusion_now == 'add':
                merge = layers.add(inputs=[part_a.output, part_b.output])
            elif fusion_now == 'sub':
                merge = layers.subtract(inputs=[part_a.output, part_b.output])
            elif fusion_now == 'max':
                merge = layers.maximum(inputs=[part_a.output, part_b.output])
            elif fusion_now == 'min':
                merge = layers.minimum(inputs=[part_a.output, part_b.output])

            a = merge
            print(f"merge: {merge.shape}")
            for j in range(0, len(new_layers_a_list[i])):
                a = new_layers_a_list[i][j](a)

            for j in range(0, len(new_layers_b_list[i])):
                b = new_layers_b_list[i][j](b)
        x = output_layer(a)
        model_new = models.Model(inputs=[part_a.input, part_b.input], outputs=x)
        model_new.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer='adam',
                          metrics=['accuracy'])
        # print(model_new.summary())
        return model_new
    else:
        print(correctNess)
        return


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

testFusion()

model = test.make_baseline_model2((112, 112, 3), activation3='softmax', conv_layers=3)
# for layer in model.layers:
#     if isinstance(layer, layers.Conv2D):
#         print("CONV!")
#     print(layer)
print(model.summary())
input()
m_new = makeSubModel(model, layers.Conv2D, -1, True)
print(m_new.summary())

"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 110, 110, 16)      448       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 108, 108, 16)      2320      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 54, 54, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 52, 52, 16)        2320      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 26, 26, 16)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 24, 24, 16)        2320      
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 12, 12, 16)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 10, 10, 16)        2320      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 5, 5, 16)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 3, 16)          2320      
_________________________________________________________________
flatten (Flatten)            (None, 144)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 60)                8700      
_________________________________________________________________
dropout (Dropout)            (None, 60)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 40)                2440      
=================================================================
"""