from tensorflow import keras
import tensorflow as tf
from tensorflow.python.keras.api._v2.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda

from tensorflow.python.keras.api._v2.keras.layers import MaxPool2D
from tensorflow.python.keras.api._v2.keras.layers import Input
from tensorflow.python.keras.api._v2.keras.models import Model
from tensorflow.python.keras.api._v2.keras.layers import Add
from tensorflow.python.keras.api._v2.keras.layers import AveragePooling2D
from tensorflow.python.keras.api._v2.keras.layers import UpSampling2D
from tensorflow.python.keras.api._v2.keras.layers import Multiply
from tensorflow.python.keras.api._v2.keras.layers import ZeroPadding2D
import numpy as np
from cnn.capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask


from cnn.rotate_methods import rotate_input, rotate_input_output_shape

from cnn.classes import siz

keras.backend.set_image_data_format('channels_last')

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
drop = 0.2


def _squash(x, axis=-1):
    """https://arxiv.org/pdf/1710.09829.pdf (eq.1)
       squash activation that normalizes vectors by their relative lengths
    """
    square_norm = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
    scale = square_norm / (1 + square_norm) / tf.sqrt(square_norm + 1e-8)
    x = tf.multiply(scale, x)
    return x



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


def cnn_maxpool(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', ke=ke2):
    global ke2

    # x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    # x = keras.layers.ReLU()(x)
    x = Dropout(drop)(x)

    # x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    # x = LocallyConnected2D(filt, ke1, strides=strides, padding='valid')(x)

    return MaxPooling2D(pool_size=maxpool, padding=padding)(x)


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
    output_soft_mask = MaxPool2D(padding='same', pool_size=(3, 3))(output_soft_mask)


    ## decoder
    skip_connections = list(reversed(skip_connections))

    # upsampling
    output_soft_mask = UpSampling2D(size=(3, 3))(output_soft_mask)
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


def get_resnet_model(num_classes: int, opt):
    global ke2

    fk = 16
    input_shape = (siz, siz, 3)
    n_class = num_classes
    routings = 3

    # ke4 = (6, 6)

    input10 = Input(shape=input_shape)
    #
    # # capsules_dims = 3
    # # num_capsules = np.prod(x.get_shape().as_list()[1:]) // capsules_dims
    # # # TensorFlow does the trick
    # # x = tf.reshape(x, [-1, num_capsules, capsules_dims])
    # # x = self._squash(x, axis=2)
    # # tf.logging.info('image after primal capsules {}'.format(x.get_shape()))
    #
    # input11 = Input(shape=input_shape)
    # input12 = Input(shape=input_shape)
    # input13 = Input(shape=input_shape)


    # x10 = ZeroPadding2D()(input10)
    # x11 = ZeroPadding2D()(input11)
    # x12 = ZeroPadding2D()(input12)
    # x13 = ZeroPadding2D()(input13)

    # x1 = keras.layers.concatenate([x10, x11], axis=1)
    # x2 = keras.layers.concatenate([x12, x13], axis=1)
    # x = keras.layers.concatenate([x1, x2], axis=2)

    x = input10

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]

    # print(x.shape)

    res1 = Conv2D(fk * 8, (3, 3), strides=(4, 4))(x)  # residual 2*2 = 4
    x = attention_layer(x, 2, input_channels=3)
    x = cnn_maxpool(fk * 2, x)  # 1

    Conv2D(fk, (1, 1), strides=(1, 1), padding='same')(x)
    x = cnn_maxpool(fk * 8, x)  # 2


    Conv2D(fk * 4, (1, 1), strides=(1, 1), padding='same')(x)
    x = Add()([x, res1])  # residual
    #
    # res2 = Conv2D(fk * 32, (3, 3), strides=(8, 8),padding='same')(x)  # residual 2*2 = 4
    # x = cnn_maxpool(fk * 8, x)  # 3
    # Conv2D(fk * 8, (1, 1), strides=(1, 1), padding='same')(x)
    # x = cnn_maxpool(fk * 16, x)  # 4
    # Conv2D(fk * 16, (1, 1), strides=(1, 1), padding='same')(x)
    # x = cnn_maxpool(fk * 32, x)  # 5
    # x = Add()([x, res2])  # residual
    #
    #
    # x = cnn_aver(fk * 64, x, padding='valid')  # 6
    # # x = cnn_aver(fk * 128, x, padding='valid')  # 7
    #
    # x = cnn_aver_loc(fk * 256, x, padding='valid')  # 1
    # x = cnn_maxpool(fk * 2, x)  # 3

    primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=fk, name='primarycap_conv2d_1')  # 7299072, 300


    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=4, dim_capsule=8,
                     routings=routings, name='digitcaps_1')(primarycaps) #opt.batchSize

    # x = keras.layers.Reshape(target_shape=[60, 60, 300])(digitcaps)
    # primarycaps = digitcaps

    # primarycaps = PrimaryCap(x, dim_capsule=800, n_channels=fk*32, name='primarycap_conv2d_2')  # 7299072, 300
    # digitcaps2 = CapsuleLayer(num_capsule=4, dim_capsule=800,
    #                          routings=routings, name='digitcaps_2')(primarycaps)  # opt.batchSize
    #
    # digitcaps = keras.layers.concatenate([digitcaps1, digitcaps2], axis=2)

    # primarycaps = PrimaryCap(x, dim_capsule=100, n_channels=fk, name='primarycap_conv2d_2')  # 7299072, 300
    # digitcaps = CapsuleLayer(num_capsule=3666, dim_capsule=100,
    #                          routings=routings, name='digitcaps_2')(primarycaps)  # opt.batchSize

    # x = keras.layers.Reshape(target_shape=[60, 60, 100])(digitcaps)
    #
    # primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=fk, name='primarycap_conv2d_3')  # 7299072, 300
    # digitcaps = CapsuleLayer(num_capsule=4, dim_capsule=8,
    #                          routings=routings, name='digitcaps_3')(primarycaps)  # opt.batchSize

    # x = digitcaps
    # x = primarycaps
    #
    # primarycaps = PrimaryCap(x, dim_capsule=8, n_channels=fk * 8)  # 7299072, 300
    #
    # digitcaps = CapsuleLayer(num_capsule=8, dim_capsule=8,
    #                          routings=routings, name='digitcaps')(primarycaps)  # opt.batchSize

    out_caps = Length(name='capsnet')(digitcaps)  #just sum(sqrt(x))- num_capsule to predict
    x = out_caps

    # print("wtf.2:", primarycaps.shape)


    # x = Flatten()(x)
    # x = keras.layers.concatenate([x,x2])

    # x = Dropout(drop)(x)

    # x = Dense(1024)(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)
    # # x = keras.layers.ReLU()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # #
    # x = Dense(50)(x)
    # x = Dropout(0.1)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)

    # x = Dense(num_classes)(x)
    #x = Activation('softmax'))
    output = Activation('softsign')(x)

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    # model = Model(inputs=[input10, input11, input12, input13], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model