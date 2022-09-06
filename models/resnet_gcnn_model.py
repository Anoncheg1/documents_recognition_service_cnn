from tensorflow import keras
from tensorflow.python.keras.api._v2.keras.layers import Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda

from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Average
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Multiply
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import LeakyReLU

from gcnn.convolutional import GConv2D
from gcnn.normalization import BatchNormalization
from gcnn.pooling import GroupPool


from classes import siz

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
drop = 0.15


# def crop_to_fit(main, to_crop):
#     from tensorflow.python.keras.api._v2.keras.layers import Cropping2D
#     from tensorflow.python.keras.api._v2.keras import backend as K
#     cropped_skip = to_crop
#     skip_size = K.int_shape(cropped_skip)[1]
#     out_size = K.int_shape(main)[1]
#     if skip_size > out_size:
#         size_diff = (skip_size - out_size) // 2
#         size_diff_odd = ((skip_size - out_size) // 2) + ((skip_size - out_size) % 2)
#         cropped_skip = Cropping2D(((size_diff, size_diff_odd),) * 2)(cropped_skip)
#     return cropped_skip

def cnn_aver_loc(filt: int, x, strides=(1,1), maxpool=(2,2), padding='valid'):
    global ke2
    x = Dropout(drop)(x)
    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    return GlobalAveragePooling2D()(x)


def cnn_aver(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', ke=ke2):
    global ke2

    x = Dropout(drop)(x)

    x = GConv2D(filt, ke, strides=strides, padding=padding, h_input='C4', h_output='C4', transpose=True)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = GConv2D(filt, ke, strides=strides, padding=padding, h_input='C4', h_output='C4', transpose=True)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = LocallyConnected2D(filt, ke1, strides=strides, padding='valid')(x)

    return GlobalAveragePooling2D()(x)


def attention_layer(inp, r=3, input_channels=2):
    skip_connections = []
    output_soft_mask = inp
    #encoder
    skip_connections.append(output_soft_mask)
    output_soft_mask = MaxPool2D(padding='same', pool_size=(2,2))(output_soft_mask)
    skip_connections.append(output_soft_mask)
    output_soft_mask = MaxPool2D(padding='same', pool_size=(2, 2))(output_soft_mask)

    # output_soft_mask =keras.layers.ZeroPadding2D(padding=1)(output_soft_mask) #add 1
    skip_connections.append(output_soft_mask)
    output_soft_mask = MaxPool2D(padding='same', pool_size=(2, 2))(output_soft_mask)


    ## decoder
    skip_connections = list(reversed(skip_connections))

    # upsampling
    output_soft_mask = UpSampling2D(size=(2, 2))(output_soft_mask)
    output_soft_mask = Add()([output_soft_mask, skip_connections[0]])
    # output_soft_mask = keras.layers.Cropping2D(cropping=(1,1))(output_soft_mask) #remove 1
    output_soft_mask = UpSampling2D(size=(2, 2))(output_soft_mask)
    output_soft_mask = Add()([output_soft_mask, skip_connections[1]])
    output_soft_mask = UpSampling2D(size=(2, 2))(output_soft_mask)
    output_soft_mask = Add()([output_soft_mask, skip_connections[2]])

    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Conv2D(input_channels, (1, 1))(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)
    # output_soft_mask = Lambda(lambda x: x + 1)(output_soft_mask)
    return Multiply()([output_soft_mask, inp])


def gnn_maxpool(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', ke=ke2):
    global ke2

    # x = GConv2D(filt, ke1, strides=strides, padding=padding, h_input='C4', h_output='C4', transpose=False)(x)
    x = Dropout(drop)(x)

    x = GConv2D(filt, ke2, strides=strides, padding=padding, h_input='C4', h_output='C4', transpose=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = keras.layers.ReLU()(x)

    x = GConv2D(filt, ke2, strides=strides, padding=padding, h_input='C4', h_output='C4', transpose=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = LocallyConnected2D(filt, ke1, strides=strides, padding='valid')(x)

    return AveragePooling2D(pool_size=maxpool, padding=padding)(x)


def get_model(num_classes: int, opt):
    global ke2

    fk = 4

    # ke4 = (6, 6)

    input10 = Input(shape=(siz, siz, 1))
    x = input10

    x = attention_layer(x, 3, input_channels=1)
    x = GConv2D(fk, (2,2), strides=(1,1), padding='same', h_input='Z2', h_output='C4', transpose=True)(x)
    x = Dropout(drop)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)


    # x = keras.layers.ReLU()(x)

    # x = keras.layers.ReLU()(x)
    #
    x = gnn_maxpool(fk * 2, x)  # 1

    res1 = GConv2D(fk * 16, (7, 7), strides=(8, 8), padding='same', h_input='C4', h_output='C4', transpose=False)(x)
    res1 = Dropout(drop)(res1)
    res1 = BatchNormalization()(res1)
    res1 = keras.layers.LeakyReLU(alpha=0.3)(res1)
    # Conv2D(fk * 2, (1, 1), strides=(1, 1), padding='same')(x)
    x = gnn_maxpool(fk * 4, x)  # 2
    # Conv2D(fk * 4, (1, 1), strides=(1, 1), padding='same')(x)

    x = gnn_maxpool(fk * 8, x)  # 3

    # Conv2D(fk * 8, (1, 1), strides=(1, 1), padding='same')(x)
    x = gnn_maxpool(fk * 16, x, padding='same')  # 4

    x = Add()([x, res1])  # residual
    res2 = GConv2D(fk * 16, (7, 7), strides=(8, 8), padding='same', h_input='C4', h_output='C4', transpose=False)(x)
    res2 = Dropout(drop)(res2)
    res2 = BatchNormalization()(res2)
    res2 = keras.layers.LeakyReLU(alpha=0.3)(res2)
    # Conv2D(fk * 16, (1, 1), strides=(1, 1), padding='same')(x)
    x = gnn_maxpool(fk * 32, x, padding='same')  # 5

    x = gnn_maxpool(fk * 64, x, padding='same')  # 6
    x = gnn_maxpool(fk * 128, x, padding='same')  # 7
    # x = Average()([x, res2])  # residual
    x = concatenate([x, res2], axis=-1)
    # x = gnn_maxpool(fk * 80, x, padding='valid')  # 7
    # x = gnn_maxpool(fk * 100, x, padding='valid')  # 8

    # x = cnn_aver(fk * 128, x, padding='valid')

    # x = cnn_aver_loc(fk * 64, x, padding='valid')

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    # x = keras.layers.concatenate([x,x2])

    # x = Dropout(drop)(x)

    # x = Dense(num_classes)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    #
    # x = Dense(256, activation='tanh')(x)
    # x = Dropout(0.4)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # #
    # x = Dense(100)(x)
    # x = Dropout(0.1)(x)
    # x = BatchNormalization()(x)
    # # x = keras.layers.ReLU(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = Dense(num_classes)(x)
    # x = Activation('softmax'))
    output = Activation('softsign')(x)

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model