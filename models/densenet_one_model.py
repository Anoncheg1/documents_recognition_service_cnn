from tensorflow import keras
from tensorflow.python.keras.api._v2.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda

import tensorflow as tf

from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Cropping2D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Average
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Multiply
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras import backend as K

from classes import siz

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
drop = 0.4
fk = 27


def crop_to_fit(main, to_crop):
    cropped_skip = to_crop
    skip_size = K.int_shape(cropped_skip)[1]
    out_size = K.int_shape(main)[1]
    if skip_size > out_size:
        size_diff = (skip_size - out_size) // 2
        size_diff_odd = ((skip_size - out_size) // 2) + ((skip_size - out_size) % 2)
        cropped_skip = Cropping2D(((size_diff, size_diff_odd),) * 2)(cropped_skip)
    return cropped_skip



def cnn_aver_loc(filt: int, x, strides=(1,1), maxpool=(2,2), padding='valid'):
    global ke2

    x = Dropout(drop)(x)

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    return AveragePooling2D(pool_size=maxpool, padding=padding)(x)


def cnn_aver(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', ke=ke2):
    global ke2

    x = Dropout(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # x = LocallyConnected2D(filt, ke1, strides=strides, padding='valid')(x)

    return AveragePooling2D(pool_size=maxpool, padding=padding)(x)


def cnn_maxpool(filt: int, x, strides=(1, 1), maxpool=(2, 2), padding='same', ke=ke2):
    global ke2



    x = Dropout(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return MaxPooling2D(pool_size=maxpool, padding=padding)(x)


growth_rate = 2
nb_filter = 2


def __dense_block(x):
    global nb_filter
    x_list = [x]
    for i in range(2):
        cb = __conv_block(x)
        x_list.append(cb)
        x = concatenate([crop_to_fit(cb, x), cb], axis=-1)
        nb_filter += growth_rate
    return x

def __conv_block(x):
    return x



def get_model(num_classes: int, opt):
    global ke2



    # ke4 = (6, 6)

    input10 = Input(shape=(siz, siz, 3))

    x = input10

    skip_list = []
    nb_dense_block = 4
    for block_idx in range(nb_dense_block):
        x = __dense_block(x)
        skip_list.append(x)

    # The last dense_block does not have a transition_down_block
    # return the concatenated feature maps without the concatenation of the input
    _, concat_list = __dense_block(x, return_concat_list=True)

    skip_list = skip_list[::-1]  # reverse the skip list

    for block_idx in range(nb_dense_block):
        l = concatenate(concat_list[1:], axis=-1)
        UpSampling2D(l)
    #
    x = Conv2D(fk, ke1, strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    res1 = Conv2D(fk * 4, (7, 7), strides=(8, 8))(x)  # residual 2*2 = 4
    x = cnn_aver(fk, x)  # 1
    # x = Conv2D(fk * 2, (1, 1), strides=(1, 1), padding='same')(x)
    # x = Dropout(drop)(x)
    x = cnn_aver(fk * 2, x)  # 2
    # x = Conv2D(fk * 4, (1, 1), strides=(1, 1), padding='same')(x)
    # x = Dropout(drop)(x)
    x = cnn_aver(fk * 4, x)  # 3
    x = Average()([x, res1])  # residual

    res2 = Conv2D(fk * 32, (7, 7), strides=(8, 8), padding='same')(x)  # residual 2*2 = 4

    # x = attention_layer(x, fk * 8, r=3)
    # x = Conv2D(fk * 8, (1, 1), strides=(1, 1), padding='same')(x)
    # x = Dropout(drop)(x)
    x = cnn_aver(fk * 8, x)  # 4
    # x = attention_layer(x, fk * 16, r=2)
    # x = Conv2D(fk * 16, (1, 1), strides=(1, 1), padding='same')(x)
    # x = Dropout(drop)(x)
    x = cnn_aver(fk * 16, x)  # 5

    x = cnn_aver(fk * 32, x, padding='same')  # 6

    x = Average()([x, res2])  # residual

    x = cnn_aver(fk * 64, x, padding='valid')  # 7

    x = Flatten()(x)
    x = Dropout(0.5)(x)

    # x = keras.layers.concatenate([x,x2])

    # x = Dropout(drop)(x)
    #

    # #
    # x = Dense(50)(x)
    # x = Dropout(0.1)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)

    x = Dense(num_classes)(x)
    output = Activation('softsign')(x)

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=[input10], outputs=output)
    return model