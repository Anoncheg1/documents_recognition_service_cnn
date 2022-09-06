from tensorflow import keras
from tensorflow.python.keras.api._v2.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda

from tensorflow.python.keras.api._v2.keras.layers import Input
from tensorflow.python.keras.api._v2.keras.models import Model
from tensorflow.python.keras.api._v2.keras.layers import Add
from tensorflow.python.keras.api._v2.keras.layers import AveragePooling2D



from cnn.rotate_methods import rotate_input, rotate_input_output_shape

from cnn.classes import siz

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)

def cnn_maxpool_loc(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same'):
    global ke2

    x = LocallyConnected2D(filt, ke1, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = LocallyConnected2D(filt, ke2, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return AveragePooling2D(pool_size=maxpool, padding=padding)(x)


def cnn_maxpool(filt: int, x, strides=(1,1), maxpool=(2,2), padding='same', dilation_rate=(1,1), ke=ke2):
    global ke2

    x = Conv2D(filt, ke, strides=strides, padding=padding, dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = Conv2D(filt, ke, padding=padding, dilation_rate=(dilation_rate[0], dilation_rate[1]))(x)
    x = BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    return MaxPooling2D(pool_size=maxpool, padding=padding)(x)
def get_resnet_model(num_classes: int):
    global ke2

    fk = 8
    input10 = Input(shape=(siz, siz, 1))
    input11 = Input(shape=(siz, siz, 1))
    input12 = Input(shape=(siz, siz, 1))
    input13 = Input(shape=(siz, siz, 1))
    # input20 = Input(shape=(siz, siz, 1)) #mask
    # input21 = Input(shape=(siz, siz, 1))
    # input22 = Input(shape=(siz, siz, 1))
    # input23 = Input(shape=(siz, siz, 1))

    # x0 = Lambda(rotate_input, output_shape=rotate_input_output_shape)([input10, input20])
    # x1 = Lambda(rotate_input, output_shape=rotate_input_output_shape)([input11, input21])
    # x2 = Lambda(rotate_input, output_shape=rotate_input_output_shape)([input12, input22])
    # x3 = Lambda(rotate_input, output_shape=rotate_input_output_shape)([input13, input23])

    # x0 = cnn_maxpool(16, x0)
    # x1 = cnn_maxpool(16, x1)
    # x2 = cnn_maxpool(16, x2)
    # x3 = cnn_maxpool(16, x3)

    x0 = cnn_maxpool(fk, input10, strides=(2,2), ke=ke3)#, dilation_rate=(2,2))  # 1
    x1 = cnn_maxpool(fk, input11, strides=(2,2), ke=ke3)#, dilation_rate=(2,2))
    x2 = cnn_maxpool(fk, input12, strides=(2,2), ke=ke3)#, dilation_rate=(2,2))
    x3 = cnn_maxpool(fk, input13, strides=(2,2), ke=ke3)#, dilation_rate=(2,2))

    res10 = Conv2D(fk*4, (3, 3), strides=(4, 4))(x0)  # residual 2*2 = 4
    res11 = Conv2D(fk*4, (3, 3), strides=(4, 4))(x1)  # residual 2*2 = 4
    res12 = Conv2D(fk*4, (3, 3), strides=(4, 4))(x2)  # residual 2*2 = 4
    res13 = Conv2D(fk*4, (3, 3), strides=(4, 4))(x3)  # residual 2*2 = 4

    x0 = cnn_maxpool(fk*2, x0, padding='same')  # 2
    x1 = cnn_maxpool(fk*2, x1, padding='same')
    x2 = cnn_maxpool(fk*2, x2, padding='same')
    x3 = cnn_maxpool(fk*2, x3, padding='same')

    res20 = Conv2D(fk * 8, (3, 3), strides=(4, 4), padding='same')(x0)  # residual 2*2 = 4
    res21 = Conv2D(fk * 8, (3, 3), strides=(4, 4), padding='same')(x1)  # residual 2*2 = 4
    res22 = Conv2D(fk * 8, (3, 3), strides=(4, 4), padding='same')(x2)  # residual 2*2 = 4
    res23 = Conv2D(fk * 8, (3, 3), strides=(4, 4), padding='same')(x3)  # residual 2*2 = 4

    x0 = cnn_maxpool(fk*4, x0, padding='same')  # 2 -64
    x1 = cnn_maxpool(fk*4, x1, padding='same')
    x2 = cnn_maxpool(fk*4, x2, padding='same')
    x3 = cnn_maxpool(fk*4, x3, padding='same')

    x0 = Add()([x0, res10])  # residual
    x1 = Add()([x1, res11])  # residual
    x2 = Add()([x2, res12])  # residual
    x3 = Add()([x3, res13])  # residual

    # ---- active cnn
    x0 = cnn_maxpool(fk*8, x0, strides=(1,1), maxpool=(2,2), padding='same')  # 3
    x1 = cnn_maxpool(fk*8, x1, strides=(1,1), maxpool=(2,2), padding='same')
    x2 = cnn_maxpool(fk*8, x2, strides=(1,1), maxpool=(2,2), padding='same')
    x3 = cnn_maxpool(fk*8, x3, strides=(1,1), maxpool=(2,2), padding='same')

    # res30 = Conv2D(fk*16, (3, 3), strides=(4, 4), padding='same')(x0res)  # residual 2*2 = 4
    # res31 = Conv2D(fk*16, (3, 3), strides=(4, 4), padding='same')(x0res)  # residual 2*2 = 4
    # res32 = Conv2D(fk*16, (3, 3), strides=(4, 4), padding='same')(x0res)  # residual 2*2 = 4
    # res33 = Conv2D(fk*16, (3, 3), strides=(4, 4), padding='same')(x0res)  # residual 2*2 = 4

    x0 = Add()([x0, res20])  # residual
    x1 = Add()([x1, res21])  # residual
    x2 = Add()([x2, res22])  # residual
    x3 = Add()([x3, res23])  # residual

    x0 = cnn_maxpool(fk*16, x0, strides=(1, 1), maxpool=(2, 2), padding='valid')  # 4
    x1 = cnn_maxpool(fk*16, x1, strides=(1, 1), maxpool=(2, 2), padding='valid')
    x2 = cnn_maxpool(fk*16, x2, strides=(1, 1), maxpool=(2, 2), padding='valid')
    x3 = cnn_maxpool(fk*16, x3, strides=(1, 1), maxpool=(2, 2), padding='valid')

    # x0 = keras.layers.multiply(x0 * x1)
    # x1 = keras.layers.multiply(x0 * x1)


    # x0res = BatchNormalization()(x0res)
    # x1res = BatchNormalization()(x1res)
    # x2res = BatchNormalization()(x2res)
    # x3res = BatchNormalization()(x3res)

    # x0res = cnn_maxpool(fk*16, x0res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x1res = cnn_maxpool(fk*16, x1res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x2res = cnn_maxpool(fk*16, x2res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x3res = cnn_maxpool(fk*16, x3res, strides=(1, 1), maxpool=(2, 2), padding='same')
    #
    # x0res = Add()([x0res, res30])  # residual
    # x1res = Add()([x1res, res31])  # residual
    # x2res = Add()([x2res, res32])  # residual
    # x3res = Add()([x3res, res33])  # residual

    # x0res = cnn_maxpool(256, x0res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x1res = cnn_maxpool(256, x1res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x2res = cnn_maxpool(256, x2res, strides=(1, 1), maxpool=(2, 2), padding='same')
    # x3res = cnn_maxpool(256, x3res, strides=(1, 1), maxpool=(2, 2), padding='same')

    x0 = cnn_maxpool_loc(fk*32, x0, strides=(1, 1), maxpool=(2, 2), padding='valid')  # 5
    x1 = cnn_maxpool_loc(fk*32, x1, strides=(1, 1), maxpool=(2, 2), padding='valid')
    x2 = cnn_maxpool_loc(fk*32, x2, strides=(1, 1), maxpool=(2, 2), padding='valid')
    x3 = cnn_maxpool_loc(fk*32, x3, strides=(1, 1), maxpool=(2, 2), padding='valid')

    x0 = Lambda(lambda x0: x0 + 5)(x0)
    x1 = Lambda(lambda x1: x1 - 5)(x1)
    x2 = Lambda(lambda x2: x2 + 15)(x2)
    x3 = Lambda(lambda x3: x3 - 15)(x3)
    x = keras.layers.concatenate([x0, x1, x2, x3], axis=1)


    x = Flatten()(x)
    # x = keras.layers.concatenate([x,xres])

    # x = Dropout(0.1)(x)

    x = Dense(200)(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.3)(x)
    # print(x)

    # x = Dense(1

    # x = Dense(800)(x)
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