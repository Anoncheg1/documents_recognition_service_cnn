from tensorflow_core.python.keras.layers.pooling import MaxPool2D
from tensorflow_core.python.keras.layers.pooling import AveragePooling2D
from tensorflow_core.python.keras.engine.input_layer import Input

from tensorflow_core.python.keras.layers.core import Dropout
from tensorflow_core.python.keras.layers.core import Flatten
from tensorflow_core.python.keras.layers.core import Lambda
from tensorflow_core.python.keras.layers.core import Dense
from tensorflow_core.python.keras.layers.core import Activation
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.layers.merge import Add
from tensorflow_core.python.keras.layers.merge import Average
from tensorflow_core.python.keras.layers.merge import Multiply
from tensorflow_core.python.keras.layers.merge import Concatenate
from tensorflow_core.python.keras.layers.noise import GaussianNoise

from tensorflow_core.python.keras.layers.convolutional import UpSampling2D
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.local import LocallyConnected2D
from tensorflow_core.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow_core.python.keras.layers.advanced_activations import ReLU
from tensorflow_core.python.keras.activations import tanh


from tensorflow_core.python.keras.layers.normalization_v2 import BatchNormalization

from classes import siz

siz = siz // 2 // 2  # 144

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
fk = 24


def cnn_aver(filt: int, x, strides=(1, 1), maxpool=(2, 2), padding='same', reg=None, drop=0, pred=0):
    global ke2

    x = Dropout(drop)(x)
    # x = GaussianNoise(drop)(x)
    res = Lambda(lambda x: x * 0.3)(x)

    x = Conv2D(pred, ke2, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # #1x1
    # x = Conv2D(pred, (1, 1), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)
    #
    # x2 = Conv2D(pred//4, (1, 2), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # x2 = BatchNormalization()(x2)
    # x2 = LeakyReLU(alpha=0.3)(x2)
    #
    # # x1 = Conv2D(pred // 4, (2, 2), strides=strides, padding=padding, activity_regularizer=reg)(x2)
    # # x1 = BatchNormalization()(x1)
    # # x1 = LeakyReLU(alpha=0.3)(x1)
    # # x = GaussianNoise(drop)(x)
    #
    # x3 = Conv2D(pred//4, (2, 1), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # x3 = BatchNormalization()(x3)
    # x3 = LeakyReLU(alpha=0.3)(x3)
    #
    # # x4 = Conv2D(pred // 4, (2, 2), strides=strides, padding=padding, activity_regularizer=reg)(x3)
    # # x4 = BatchNormalization()(x4)
    # # x4 = LeakyReLU(alpha=0.3)(x4)
    #
    # # 1x1
    # # x4 = Conv2D(pred // 4, (1, 1), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # # x4 = BatchNormalization()(x4)
    # # x4 = LeakyReLU(alpha=0.3)(x4)
    #
    # x5 = Conv2D(pred // 4, (1, 3), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # x5 = BatchNormalization()(x5)
    # x5 = LeakyReLU(alpha=0.3)(x5)
    #
    # x6 = Conv2D(pred // 4, (3, 1), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # x6 = BatchNormalization()(x6)
    # x6 = LeakyReLU(alpha=0.3)(x6)
    # # x = GaussianNoise(drop)(x)
    #
    # x = Concatenate()([x2, x3, x5, x6])

    # # 1x1 to correct depth
    # x = Conv2D(pred, (1, 1), strides=strides, padding=padding, activity_regularizer=reg)(x)
    # # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.3)(x)

    x = Average()([x, res])  # residual

    # x = AveragePooling2D(pool_size=maxpool, padding=padding)(x)
    # ---------------- Pooling part ----------
    x = Dropout(0.1)(x)

    x = Conv2D(filt, ke2, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # x1 = Conv2D(filt // 2, (3, 3), strides=ke2, padding=padding, activity_regularizer=reg)(x)
    # x1 = BatchNormalization()(x1)
    # x1 = LeakyReLU(alpha=0.3)(x1)

    # x2 = MaxPool2D(pool_size=maxpool, padding=padding)(x)
    # x2 = Conv2D(filt // 2, (1, 1), strides=strides, padding=padding, activity_regularizer=reg)(x2)
    # x2 = BatchNormalization()(x2)
    # x2 = LeakyReLU(alpha=0.3)(x2)

    x = AveragePooling2D(pool_size=maxpool, padding=padding)(x)
    # x3 = Conv2D(filt // 2, (1, 1), strides=strides, padding=padding, activity_regularizer=reg)(x3)
    # x3 = BatchNormalization()(x3)
    # x3 = LeakyReLU(alpha=0.3)(x3)

    # x = Concatenate()([x1, x2])

    output = x
    print(output.shape)
    return output


def cnn_aver_l(filt: int, x, strides=(1, 1), maxpool=(2, 2), padding='same', ke=ke2, reg=None, drop=0):
    global ke2

    x = Dropout(drop)(x)
    # x = GaussianNoise(drop)(x)

    x = LocallyConnected2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = GaussianNoise(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    output = AveragePooling2D(pool_size=maxpool, padding=padding)(x)
    print(output.shape)
    return output


def attention_layer(inp, input_channels=2, r=4, drop=0):

    inp = Dropout(drop)(inp)
    # inp = GaussianNoise(drop)(inp)

    output_soft_mask = inp

    skip_connections = []
    #encoder
    for i in range(r):
        # output_soft_mask = Conv2D(input_channels, ke1, padding='same')(output_soft_mask)
        # # x = BatchNormalization()(x)
        # output_soft_mask = LeakyReLU(alpha=0.5)(output_soft_mask)
        skip_connections.append(output_soft_mask)
        output_soft_mask = MaxPool2D(padding='same', pool_size=(2, 2))(output_soft_mask)
    # output_soft_mask =keras.layers.ZeroPadding2D(padding=1)(output_soft_mask) #add 1



    ## decoder
    skip_connections = list(reversed(skip_connections))

    # upsampling
    for i in range(r):
        x = Conv2D(input_channels, ke2, padding='same')(output_soft_mask)
        x = BatchNormalization()(x)
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

    # x = Dropout(opt.drop)(x)
    # x = GaussianNoise(opt.drop+0.4)(x)

    #
    x = Conv2D(fk, ke1, strides=(1,1))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(opt.drop)(x)
    #
    x2 = x
    # x3 = x
    # x4 = x
    # ---- ENSAMBLE

    def ensamble(xx):
        xx = attention_layer(xx, fk, r=4)
        xx = cnn_aver(fk, xx, drop=opt.drop, pred=fk)  # 1
        # xx = attention_layer(xx, fk, r=3)
        xx = cnn_aver(fk * 2, xx, drop=opt.drop, pred=fk)  # 2
        # x = attention_layer(x, fk*2, r=4)
        xx = cnn_aver(fk * 4, xx, drop=opt.drop, pred=fk * 2)  # 3
        xx = cnn_aver(fk * 8, xx, drop=opt.drop, pred=fk * 4)  # 4
        xx = cnn_aver(fk * 16, xx, drop=opt.drop, pred=fk * 8)  # 5
        # xx = cnn_aver(fk * 32, xx, drop=opt.drop, pred=fk * 16)  # 6

        # xx = cnn_aver_l(fk * 8, xx, drop=opt.drop, padding='valid')  # 4
        # xx = cnn_aver_l(fk * 16, xx, drop=opt.drop, padding='valid')  # 5

        xx = Flatten()(xx)
        # xx = Dropout(opt.drop)(xx)
        # xx = BatchNormalization()(xx)

        xx = LeakyReLU(alpha=0.3)(xx)
        # xx = ReLU()(xx)
        # xx = tanh(xx)
        xx = Dense(4000)(xx)

        # xx = Dropout(opt.drop + 0.5)(xx)
        xx = BatchNormalization()(xx)
        xx = ReLU()(xx)  # must be 0-1 becouse of loss function
        # xx = Dense(num_classes, activation='softmax')(xx)
        # xx = Dropout(0.7)(xx)
        # xx = Activation('softmax')(xx)
        xx = Dense(128)(xx)



        # # xx = LeakyReLU(alpha=0.3)(xx)
        # # xx = Dropout(opt.drop)(xx)
        #
        # # xx = Dropout(opt.drop)(xx)
        # xx = LocallyConnected2D(2000, ke1, strides=ke1, padding='valid', activation=None)(xx)
        # xx = Dropout(0.8)(xx)
        # # xx = BatchNormalization()(xx)
        # xx = LeakyReLU(alpha=0.5)(xx)
        #
        # xx = Conv2D(2, ke1, strides=ke1, padding='valid', activation=None)(xx)
        # # xx = Dropout(0.8)(xx)
        # xx = BatchNormalization()(xx)

        # xx = Flatten()(xx)
        # xx = tanh(xx)


        # xx = LeakyReLU(alpha=0.7)(xx)
        # xx = Dense(num_classes * 4)(xx)
        return xx

    # ----------------- FIRST
    x = ensamble(x)
    # ----------------- SECOND
    x2 = ensamble(x2)

    # # ----------------- THIRD
    # x3 = ensamble(x3)
    # # ----------------- FORTH
    # x4 = ensamble(x4)
    #
    # x = Average()([x, x2, x3, x4])
    x = Average()([x, x2])

    # x = Dense(512)(x)
    # x = Dropout(0.7)( x)
    # x = ReLU()(x)

    # x = Dense(64)(x)
    # x = Dropout(opt.drop)(x)
    # x = Dense(1024)(x)
    # # x = BatchNormalization()(x)
    # x = Dense(512)(x)
    # x = Dropout(opt.drop+0)(x)

    # x = LeakyReLU(alpha=1)(x)
    # x = Flatten()(x)

    # output = Dense(num_classes, activation='softsign')(x)
    output = Dense(num_classes, activation='softmax')(x)
    # output = x

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model
