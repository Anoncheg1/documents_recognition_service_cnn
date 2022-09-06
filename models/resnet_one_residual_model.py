from tensorflow import keras
from tensorflow.python.keras.api._v2.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda

from tensorflow.python.keras.api._v2.keras.layers import Input
from tensorflow.python.keras.api._v2.keras.models import Model
from tensorflow.python.keras.api._v2.keras.layers import Add
from tensorflow.python.keras.api._v2.keras.layers import AveragePooling2D
from tensorflow.python.keras.api._v2.keras.layers import UpSampling2D



from cnn.rotate_methods import rotate_input, rotate_input_output_shape

from cnn.classes import siz

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)

def cnn_aver_loc(filt: int, x, strides=(1,1), maxpool=(2,2), padding='valid'):
    global ke2

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return AveragePooling2D(pool_size=maxpool, padding=padding)(x)


def cnn_maxpool(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', ke=ke2):
    global ke2

    # x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    # x = keras.layers.ReLU()(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return MaxPooling2D(pool_size=maxpool, padding=padding)(x)

def get_resnet_model(num_classes: int):
    global ke2

    fk = 4

    # ke4 = (6, 6)

    input10 = Input(shape=(siz, siz, 1))
    input11 = Input(shape=(siz, siz, 1))
    input12 = Input(shape=(siz, siz, 1))
    input13 = Input(shape=(siz, siz, 1))
    x1 = keras.layers.concatenate([input10, input11], axis=1)
    x2 = keras.layers.concatenate([input12, input13], axis=1)
    x = keras.layers.concatenate([x1, x2], axis=2)

    # x = cnn_maxpool(fk, x, strides=(2,2), maxpool=(2,2), ke=ke3)  # 1

    res1 = Conv2D(fk * 4, (7, 7), strides=(8, 8))(x)  # residual 2*2 = 4
    x = cnn_maxpool(fk, x)  # 1
    x = cnn_maxpool(fk * 2, x)  # 1
    # x = Dropout(0.1)(x)
    res2 = Conv2D(fk * 8, (3, 3), strides=(4, 4))(x)  # residual 2*2 = 4
    x = cnn_maxpool(fk * 4, x)  # 1
    # x = Dropout(0.15)(x)

    x = Add()([x, res1])  # residual
    res3 = Conv2D(fk * 16, (3, 3), strides=(4, 4))(x)  # residual 2*2 = 4
    x = cnn_maxpool(fk * 8, x)  # 1
    # x = Dropout(0.15)(x)

    # cop1 = Conv2D(fk * 64, (1, 1), strides=(1, 1))(x)

    x = Add()([x, res2])  # residual
    res4 = Conv2D(fk * 32, (3, 3), strides=(4, 4), padding='same')(x)  # residual 2*2 = 4
    x = cnn_maxpool(fk * 16, x)  # 1
    x = Dropout(0.1)(x)
    x = Add()([x, res3])  # residual
    # res5 = Conv2D(fk * 64, (3, 3), strides=(4, 4), padding='same')(x)  # residual 2*2 = 4
    x = cnn_maxpool(fk * 32, x)  # 1
    x = Dropout(0.1)(x)
    x = Add()([x, res4])  # residual
    x = cnn_maxpool(fk * 64, x)  # 1
    x = cnn_maxpool(fk * 64, x)  # 1
    x = Dropout(0.1)(x)
    x = cnn_aver_loc(fk * 128, x)  # 1



    # xup = UpSampling2D((6,6))(x)
    # xup = Lambda(lambda xup: xup + 2)(xup)
    # x2 = Add()([cop1, xup])
    # x2 = cnn_maxpool(fk * 64, x2)  # 1

    # x = Add()([x, res5])  # residual
    # x = cnn_maxpool(fk * 128, x)  # 1
    # x = cnn_aver_loc(fk * 128, x)  # 1
    # x = keras.layers.GlobalAveragePooling2D()(x)
    # x2 = Flatten()(x2)
    x = Flatten()(x)
    # x = keras.layers.concatenate([x,x2])

    x = Dropout(0.1)(x)
    # # x = keras.layers.concatenate([x,xres])

    x = Dense(200)(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)
    # x = keras.layers.ReLU()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    #
    # x = Dense(128)(x)
    # x = Dropout(0.3)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # # x = keras.layers.ReLU()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    #
    # x = Dense(100)(x)
    # x = Dropout(0.2)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)
    #
    # x = Dense(1200)(x)
    # x = Dropout(0.6)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)

    # x = Dense(100))
    # x = Dropout(0.1))
    # x = BatchNormalization())
    # x = Activation('relu'))

    # x = Dropout(0.1))
    x = Dense(num_classes)(x)
    #x = Activation('softmax'))
    output = Activation('softsign')(x)

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=[input10, input11, input12, input13], outputs=output)
    return model