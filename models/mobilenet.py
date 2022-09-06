from tensorflow.python.keras.layers import MaxPool2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LocallyConnected2D
from tensorflow.python.keras.layers import BatchNormalizationV2
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Add
from tensorflow.python.keras.layers import Average
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import Multiply
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import ReLU
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import Lambda
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.activations import softsign
# from tensorflow.python.keras.regularizers import l
from tensorflow.python.keras.api._v2.keras.layers import ZeroPadding2D

from classes import siz

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
ke4 = (3, 3)
fk = 36


def cnn_aver(filt: int, x, strides=(2, 2), maxpool=(2, 2), padding='same', ke=ke2, reg=None, drop=0, max=0):
    global ke2

    x = Dropout(drop)(x)

    h1 = Conv2D(filt//2, ke, strides=(1,1), padding=padding, activity_regularizer=reg)(x)
    h1 = BatchNormalizationV2()(h1)
    h1 = LeakyReLU(alpha=0.3)(h1)
    #
    h2 = Conv2D(filt//2, ke, strides=(1, 1), padding=padding, activity_regularizer=reg)(h1)
    h2 = BatchNormalizationV2()(h2)
    h2 = LeakyReLU(alpha=0.3)(h2)
    #
    # h3 = Conv2D(filt // 2, ke, strides=(1, 1), padding=padding, activity_regularizer=reg)(h2)
    # h3 = BatchNormalizationV2()(h3)
    # h3 = LeakyReLU(alpha=0.3)(h3)

    # h4 = Conv2D(filt // 5, ke4, strides=strides, padding=padding, activity_regularizer=reg)(x)
    # h4 = BatchNormalizationV2()(h4)
    # h4 = LeakyReLU(alpha=0.3)(h4)

    # h5 = Conv2D(filt // 5, ke4, strides=strides, padding=padding, activity_regularizer=reg)(x)
    # if max == 0:
    #     h5 = Conv2D(filt // 2, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    #     h5 = BatchNormalizationV2()(h5)
    #     h5 = LeakyReLU(alpha=0.3)(h5)
    # elif max == 1:
    #     h5 = MaxPool2D(pool_size=maxpool, padding=padding)(x)
    # else:
    #     h5 = AveragePooling2D(pool_size=maxpool, padding=padding)(x)
    h5 = concatenate([x, h2], axis=-1)
    # output = concatenate([h2, h5], axis=-1)

    if max == 0:
        output = Conv2D(filt, ke1, strides=strides, padding=padding, activity_regularizer=reg)(h5)
        output = BatchNormalizationV2()(output)
        output = LeakyReLU(alpha=0.3)(output)
    elif max == 1:
        output = MaxPool2D(pool_size=maxpool, padding=padding)(h5)
    else:
        output = AveragePooling2D(pool_size=maxpool, padding=padding)(h5)

    print(output.shape)

    return output  # here


def attention_layer(inp, input_channels=2, r=4):
    skip_connections = []
    output_soft_mask = inp
    #encoder
    for i in range(r):
        # output_soft_mask = Conv2D(input_channels, ke1, padding='same')(output_soft_mask)
        # # x = BatchNormalizationV2()(x)
        # output_soft_mask = LeakyReLU(alpha=0.5)(output_soft_mask)
        skip_connections.append(output_soft_mask)
        output_soft_mask = MaxPool2D(padding='same', pool_size=(2, 2))(output_soft_mask)
    # output_soft_mask =keras.layers.ZeroPadding2D(padding=1)(output_soft_mask) #add 1



    ## decoder
    skip_connections = list(reversed(skip_connections))

    # upsampling
    for i in range(r):
        x = Conv2D(input_channels, ke2, padding='same')(output_soft_mask)
        x = BatchNormalizationV2()(x)
        output_soft_mask = ReLU()(x)
        output_soft_mask = UpSampling2D(size=(2, 2))(output_soft_mask)
        output_soft_mask = Add()([output_soft_mask, skip_connections[i]])
    # output_soft_mask = keras.layers.Cropping2D(cropping=(1,1))(output_soft_mask) #remove 1

    output_soft_mask = Conv2D(input_channels, ke1)(output_soft_mask)
    # output_soft_mask = Conv2D(input_channels, ke1)(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)
    output_soft_mask = Lambda(lambda x: x + 1)(output_soft_mask)
    return Multiply()([output_soft_mask, inp])


def get_model(num_classes: int, opt):
    global ke2

    input10 = Input(shape=(siz, siz, 1))
    x = input10

    #
    x = Conv2D(fk//2, ke1, strides=(1,1))(x)
    # x = Dropout(opt.drop)(x)
    x = BatchNormalizationV2()(x)
    x = ReLU()(x)

    # res1 = Conv2D(fk * 4, (8, 8), strides=(8, 8))(x)  # residual 2*2 = 4
    # res1 = Dropout(opt.drop/1.5)(res1)
    # res1 = BatchNormalizationV2()(res1)
    # res1 = LeakyReLU(alpha=0.3)(res1)

    x = cnn_aver(fk, x, drop=opt.drop, max=1)  # 1
    x = attention_layer(x, fk, r=5)

    x = cnn_aver(fk * 2, x, drop=opt.drop, max=2)  # 2
    x = attention_layer(x, fk * 2, r=3)

    x = cnn_aver(fk * 4, x, drop=opt.drop, max=1)  # 3
    x = attention_layer(x, fk * 4, r=3)
    # x = Average()([x, res1])  # residual

    # res2 = Conv2D(fk * 46+108, (16, 16), strides=(16, 16), padding='same')(x)  # residual 2*2 = 4
    # res2 = Dropout(opt.drop/1.5)(res2)
    # res2 = BatchNormalizationV2()(res2)
    # res2 = LeakyReLU(alpha=0.3)(res2)

    x = cnn_aver(fk * 6, x, drop=opt.drop, max=2)  # 4
    x = attention_layer(x, 252, r=2)
    x = cnn_aver(fk * 12, x, drop=opt.drop, max=1)  # 5

    x = cnn_aver(fk * 24, x, drop=opt.drop, max=2)  # 6



    x = cnn_aver(fk * 48, x, drop=opt.drop, max=1)  # 7



    # x = Add()([x, res2])  # residual

    x = cnn_aver(fk * 86, x, drop=opt.drop, max=0)  # 8

    x = Flatten()(x)
    x = Dropout(0.9)(x)

    # #
    # x = Dense(5024, activation='tanh')(x)
    # x = Dropout(0.8)(x)
    # x = BatchNormalizationV2()(x)

    # x = Dense(1024)(x)
    # x = Dropout(0.1)(x)
    # x = BatchNormalizationV2()(x)
    # x = softsign(x)
    # x = LeakyReLU(alpha=0.3)(x)

    # x = Dense(124, activation='tanh')(x)
    # x = Dropout(0.2)(x)
    # x = BatchNormalizationV2()(x)
    # x = LeakyReLU(alpha=0.8)(x)

    output = Dense(num_classes, activation='softsign')(x)

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model
