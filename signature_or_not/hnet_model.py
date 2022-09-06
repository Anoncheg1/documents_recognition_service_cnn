import tensorflow as tf
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import GaussianNoise

from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import LocallyConnected2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU
# from tensorflow.keras.layers import tanh


from tensorflow.keras.layers import BatchNormalization

from classes import siz

siz = siz // 2 // 2  # 144

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
fk = 20


def cnn_aver(filt: int, x, strides=(1, 1), maxpool=(4, 4), padding='valid', ke=ke2, reg=None, drop=0):
    global ke2

    x = Dropout(drop)(x)
    # x = GaussianNoise(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)
    # x = GaussianNoise(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    # output = AveragePooling2D(pool_size=maxpool, padding=padding, strides=(4, 4))(x)
    output = MaxPool2D(pool_size=maxpool, padding=padding, strides=maxpool)(x)
    print(output.shape)
    return output


def get_model(num_classes: int, opt):
    global ke2

    input10 = Input(shape=(opt.size_y, opt.size_x, 1))
    x = input10

    # x = Dropout(opt.drop)(x)
    # x = GaussianNoise(opt.drop+0.4)(x)

    #
    # x = Conv2D(fk, ke1, strides=(1, 1))(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)
    # x = Dropout(opt.drop)(x)
    c = opt.size_y - opt.size_x
    print(c)

    # x = ZeroPadding2D(((c//2,c//2), (0,0)))(x)

    x2 = x
    x3 = x
    x4 = x
    x5 = x
    # ---- ENSAMBLE

    def ensamble(xx, fkl):
        # res1 = Conv2D(fk * 4, (8, 8), strides=(8, 8))(xx)  # residual 2*2 = 4
        # res1 = Dropout(opt.drop/1.5)(res1)
        # if flag1:
        #     xx = cnn_aver(fkl, xx, drop=opt.drop, maxpool=(2,2))  # 1
        # else:
        xx = cnn_aver(fkl, xx, drop=opt.drop)  # 1
        # xx = cnn_aver(fkl, xx, drop=opt.drop, maxpool=(3, 5))  # 1
        # x = attention_layer(x, fk, r=5)
        xx = cnn_aver(fkl * 2, xx, drop=opt.drop)  # 2
        # x = attention_layer(x, fk*2, r=4)
        # xx = cnn_aver(fkl * 4, xx, drop=opt.drop, maxpool=(2,2))  # 3
        xx = cnn_aver(fkl * 4, xx, drop=opt.drop)  # 3
        #     xx = cnn_aver(fk * 8, xx, drop=opt.drop)  # 4
        # x = attention_layer(x, fk*4, r=3)
        # xx = Average()([xx, res1])  # residual
        # xx = cnn_aver_l(fk * 8, xx, drop=opt.drop, padding='valid')  # 4
        # xx = cnn_aver_l(fk * 16, xx, drop=opt.drop, padding='valid')  # 5
        # # x = cnn_aver(fk * 32, x, drop=opt.drop, padding='valid')  # 5
        #
        # xx = Flatten()(xx)
        # xx = Dropout(opt.drop + 0.4)(xx)
        # xx = Dense(800)(xx)
        # xx = LeakyReLU(alpha=0.3)(xx)
        # xx = Dropout(opt.drop + 0.1)(xx)
        return xx

    def ensamble2(xx, fkl, drop=opt.drop):
        xx = cnn_aver(fkl, xx, drop=drop, maxpool=(2,2))  # 1
        xx = cnn_aver(fkl * 2, xx, drop=drop, maxpool=(2, 2))  # 2
        xx = cnn_aver(fkl * 4, xx, drop=drop, maxpool=(4, 4))  # 3
        xx = cnn_aver(fkl * 8, xx, drop=drop, maxpool=(4, 4))  # 4
        xx = Flatten()(xx)
        return xx



    # ----------------- FIRST
    x = ensamble(x, fkl=fk)
    # ----------------- SECOND
    x2 = ensamble(x2, fkl=fk)
    # ----------------- THIRD
    # x3 = ensamble(x3, fkl=fk)
    # ----------------- FORTH
    # x4 = ensamble(x4, fkl=fk)
    x4 = ensamble2(x4, fkl=fk, drop=opt.drop)
    x5 = ensamble2(x5, fkl=fk, drop=opt.drop)

    x = Average()([x, x2])
    x2 = Average()([x4, x5])
    # x = ZeroPadding2D(((0, 1),(1,0)))(x)

    # x = Reshape([-1, 2,-1])(x)
    # x = ZeroPadding1D((1,0))(x)
    x = Flatten()(x)
    # x = Dropout(opt.drop)(x)
    x = Dense(160)(x)  # , activation=tf.nn.relu

    x2 = Flatten()(x2)

    #
    x = Average()([x, x2])
    x = Dropout(opt.drop+0.1)(x)
    # x = BatchNormalization()(x)
    x = Dense(230, activation=tf.nn.relu)(x)
    output = Dense(num_classes, activation=tf.nn.sigmoid)(x)  # softsign
    # print(x4)



    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model
