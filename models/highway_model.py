import tensorflow as tf
import numpy as np
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
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalizationV2
from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.activations import tanh
from tensorflow.python.keras.layers import Multiply
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import Subtract
# from tensorflow.python.keras.initializers import constant
from tensorflow.keras.initializers import Constant
from tensorflow.python.keras.regularizers import l2, l1


from classes import siz

# pools = 5
# total_layers = 10  # Specify how deep we want our network
# units_between_stride = 2  # total_layers // pools
ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
ke4 = (4, 4)
fk = 28
# fk=30


def highwayUnitR(filt: int, input_layer, strides=(1,1),padding='same', ke=ke4, drop=0):
    input_layer = Dropout(drop)(input_layer)
    H = Conv2D(filt, ke, strides=strides, padding=padding)(input_layer)
    # H = Dropout(drop)(H)
    H = BatchNormalizationV2()(H)
    # H = ReLU()(H)  # here
    H = LeakyReLU(alpha=0.3)(H)
    H = Conv2D(filt, ke, strides=strides, padding=padding)(H)
    # H = Dropout(drop)(H)
    H = BatchNormalizationV2()(H)
    # H = ReLU()(H)  # here
    H = LeakyReLU(alpha=0.3)(H)
    T = Conv2D(filt, ke, strides=strides, padding=padding,
               # We initialize with a negative bias to push the network to use the skip connection
               bias_initializer=Constant(-1.0))(input_layer)
    # T = Dropout(drop)(T)
    T = BatchNormalizationV2()(T)
    T = ReLU()(T)
    T = Activation('sigmoid')(T)
    T = Lambda(lambda x: 1.0 - x)(T)
    input_layer = Conv2D(filt, ke1)(input_layer)  # enlarge channels count
    m = Multiply()([input_layer, T])
    output = Multiply()([H, T])
    output = Add()([output, m])
    output = AveragePooling2D(ke)(output)
    # output = concatenate([output, m], axis=-1)  # here
    return output


def highwayUnitC(filt: int, input_layer, strides=(1,1),padding='same', ke=ke4, drop=0):
    input_layer = Dropout(drop)(input_layer)
    H1 = Conv2D(filt//2, ke2, strides=strides, padding=padding)(input_layer)

    H1 = BatchNormalizationV2()(H1)
    H1 = LeakyReLU(alpha=0.3)(H1)
    H2 = Conv2D(filt//2, ke, strides=strides, padding=padding)(H1)
    # H2 = Dropout(drop)(H2)
    H2 = BatchNormalizationV2()(H2)
    H2 = LeakyReLU(alpha=0.3)(H2)
    # H3 = Conv2D(filt, ke, strides=strides, padding=padding)(input_layer)
    # H3 = BatchNormalizationV2()(H3)
    # H3 = LeakyReLU(alpha=0.3)(H3)
    # H = ReLU()(H)
    T = Conv2D(filt//2, ke, strides=strides, padding=padding,
        # We initialize with a negative bias to push the network to use the skip connection
                        bias_initializer=Constant(-1.0))(input_layer)
    T = BatchNormalizationV2()(T)
    # # T = LeakyReLU(alpha=0.3)(T)
    # T = ReLU()(T)
    T = Activation('sigmoid')(T)
    #
    # m1 = Multiply()([H1, T])
    input_layer = Conv2D(filt//2, ke1)(input_layer)  # enlarge channels count
    m1 = Multiply()([H2, T])
    T = Lambda(lambda x: 1.0 - x)(T)
    m2 = Multiply()([input_layer, T])

    output = concatenate([m1, m2])

    # m2 = Multiply()([H2, T])
    # output = Add()([m,m2, input_layer])  # here
    output = AveragePooling2D(ke)(output)
    return output


def highwayUnitSmall(filt: int, input_layer, strides=(1,1),padding='same', ke=ke4, drop=0):
    input_layer = Dropout(drop)(input_layer)
    H = Conv2D(filt, ke, strides=strides, padding=padding)(input_layer)
    # H = Dropout(drop)(H)
    H = BatchNormalizationV2()(H)
    H = LeakyReLU(alpha=0.3)(H)
    H = Conv2D(filt, ke, strides=strides, padding=padding)(H)
    # H = Dropout(drop)(H)
    H = BatchNormalizationV2()(H)
    H = LeakyReLU(alpha=0.3)(H)
    T = Conv2D(filt, ke, strides=strides, padding=padding,
        # We initialize with a negative bias to push the network to use the skip connection
                        bias_initializer=Constant(-1.0))(input_layer)
    T = BatchNormalizationV2()(T)
    T = ReLU()(T)
    T = Activation('sigmoid')(T)
    # T = Activation('softsign')(T)

    T = Lambda(lambda x: 1.0 - x)(T)
    input_layer = Conv2D(filt, ke1)(input_layer)  # enlarge channels count
    m = Multiply()([input_layer, T])
    output = Add()([H, m])  # here
    # output = concatenate([H, m], axis=-1)
    # output = AveragePooling2D(ke)(output)
    return output


def get_model(num_classes: int, opt):
    units_between_stride = 2

    input10 = Input(shape=(opt.size_y, opt.size_x, 1))

    x = Conv2D(fk//2, ke1, strides=(1,1))(input10)
    # x = Dropout(drop)(x)
    x = BatchNormalizationV2()(x)
    x = ReLU()(x)
    # x = input10


    i = 1
    x = highwayUnitR(fk * i, x, padding='same', ke=ke2, drop=opt.drop)  # 1

    res1 = Conv2D(fk*8, (16,16), strides=(16, 16))(x)
    res1 = Dropout(opt.drop/1.5)(res1)
    res1 = BatchNormalizationV2()(res1)
    res1 = LeakyReLU(alpha=0.3)(res1)
    # res1 = MaxPool2D((16,16))(x)  # residual

    i = 2
    # x = highwayUnit(fk * i, x, padding='same', ke=ke2)
    x = highwayUnitR(fk * i, x, drop=opt.drop, ke=ke4)  # 2

    i = 8
    # x = highwayUnit(fk * i, x, padding='same', ke=ke2)
    x = highwayUnitC(fk * i, x, drop=opt.drop)  # 3


    x = Add()([x, res1])  # residual
    # x = concatenate([x, res1], axis=-1)
    # res2 = MaxPool2D((4,4))(x)
    # print(x.shape)
    i = 16
    # res2 = Conv2D(fk * 32, (10,10), strides=(9,9), padding='same')(x)
    # res2 = Dropout(opt.drop / 1.5)(res2)
    # res2 = BatchNormalizationV2()(res2)
    # res2 = LeakyReLU(alpha=0.3)(res2)
    # print(res2.shape)
    # res2 = AveragePooling2D((8, 8))(x)  # residual
    # res2 = Dropout(opt.drop/1.5)(res2)

    # x = highwayUnit(fk * i, x, padding='same', ke=ke2)
    x = highwayUnitSmall(fk * i, x, drop=opt.drop)  # 4


    # i = 16
    # # # x = highwayUnit(fk * i, x, padding='same', ke=ke2)
    # x = highwayUnitSmall(fk * i, x)
    # x = AveragePooling2D((2,2))(x)

    i = 32
    x = highwayUnitSmall(fk * i, x, ke=ke2, drop=opt.drop)

    # x = Conv2D(fk * i*2, ke2, strides=(1, 1), padding='same')(x)
    # x = BatchNormalizationV2()(x)
    # x = ReLU()(x)
    # x = concatenate([x, res2], axis=-1)
    # x = Add()([x, res2])  # residual

    # x = highwayUnitSmall(fk * i, x)



    # x = GlobalAveragePooling2D()(x)

    # res2 = Conv2D(fk * 6, (3, 3), strides=(4, 4))(x)  # residual
    #
    # i = 5
    # x = Conv2D(fk * i, ke1)(x)  # enlarge channels count
    # for j in range(units_between_stride):
    #     x = highwayUnit(fk * i, x, padding='same')
    # x = AveragePooling2D((4,4))(x)
    #
    # i = 6
    # x = Conv2D(fk * i, ke1)(x)  # enlarge channels count
    # for j in range(units_between_stride):
    #     x = highwayUnit(fk * i, x, padding='same')
    # x = AveragePooling2D((4,4))(x)
    #
    # x = Average()([x, res2])  # residual

    # for i in range(1, pools + 1):
    #     if i != 1:  # first layer
    #         x = Conv2D(fk * i, ke1)(x)  # enlarge channels count
    #     for j in range(units_between_stride):
    #         x = highwayUnit(fk * i, x, padding='same')
    #     x = AveragePooling2D()(x)

    x = Flatten()(x)
    x = Dropout(opt.drop)(x)

    # x = Dense(num_classes, kernel_regularizer=l2(0.01),
    #             activity_regularizer=l1(0.01))(x)

    # x = Dense(256, activation='tanh')(x)
    # x = Dropout(0.4)(x)
    # # x = Dense(num_classes)(x)
    # x = BatchNormalizationV2()(x)
    # x = LeakyReLU(alpha=0.7)(x)

    x = Dense(num_classes)(x)
    # output = Activation('softsign')(x)
    output = Activation('sigmoid')(x)
    # output = Activation('softmax')(x)

    model = Model(inputs=input10, outputs=output)
    return model