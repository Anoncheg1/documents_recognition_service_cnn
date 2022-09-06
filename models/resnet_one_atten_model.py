from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras import Input
# from tensorflow_core.python.keras.engine.input_layer import Input

from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Average
from tensorflow.keras.layers import Multiply

from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ReLU


from tensorflow.keras.layers import BatchNormalization

ke1 = (1, 1)
ke2 = (2, 2)
ke3 = (3, 3)
fk = 12


def cnn_aver(filt: int, x, strides=(1, 1), maxpool=(2, 2), padding='same', ke=ke2, reg=None, drop=0):
    global ke2

    x = Dropout(drop)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    x = Conv2D(filt, ke, strides=strides, padding=padding, activity_regularizer=reg)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.3)(x)

    output = AveragePooling2D(pool_size=maxpool, padding=padding)(x)
    print(output.shape)
    return output


def attention_layer(inp, input_channels=2, r=4):
    skip_connections = []
    output_soft_mask = inp
    #encoder
    for i in range(r):
        output_soft_mask = Conv2D(input_channels, ke1, padding='same')(output_soft_mask)
        output_soft_mask = BatchNormalization()(output_soft_mask)
        output_soft_mask = ReLU()(output_soft_mask)
        skip_connections.append(output_soft_mask)
        output_soft_mask = MaxPool2D(padding='same', pool_size=(2, 2))(output_soft_mask)
        print(output_soft_mask.shape)
    # output_soft_mask =keras.layers.ZeroPadding2D(padding=1)(output_soft_mask) #add 1



    ## decoder
    skip_connections = list(reversed(skip_connections))

    # upsampling
    for i in range(r):
        x = Conv2D(input_channels, ke2, padding='same')(output_soft_mask)
        x = BatchNormalization()(x)
        output_soft_mask = ReLU()(x)
        output_soft_mask = UpSampling2D(size=(2, 2))(output_soft_mask)
        # print()
        added = output_soft_mask
        output_soft_mask = Add()([added, skip_connections[i]])
    # output_soft_mask = keras.layers.Cropping2D(cropping=(1,1))(output_soft_mask) #remove 1

    output_soft_mask = Conv2D(input_channels, ke1)(output_soft_mask)
    # output_soft_mask = Conv2D(input_channels, ke1)(output_soft_mask)
    output_soft_mask = Activation('sigmoid')(output_soft_mask)
    output_soft_mask = Lambda(lambda x: x + 1)(output_soft_mask)
    return Multiply()([output_soft_mask, inp])


def get_model(num_classes: int, opt):
    global ke2

    input10 = Input(shape=(opt.size_y, opt.size_x, 1))
    # input10 = ZeroPadding2D(padding=([[8, 8], [8, 8]]))(input10)
    x = input10


    #
    # x = Conv2D(fk, ke1, strides=(1,1))(x)
    # # x = Dropout(opt.drop)(x)
    # x = BatchNormalization()(x)
    # x = ReLU()(x)

    # res1 = Conv2D(fk * 4, (8, 8), strides=(8, 8))(x)  # residual 2*2 = 4
    res1 = MaxPool2D(padding='same', pool_size=(8, 8))(x)
    # res1 = Dropout(opt.drop/1.5)(res1)
    # res1 = BatchNormalization()(res1)
    # res1 = LeakyReLU(alpha=0.3)(res1)
    x = attention_layer(x, fk, r=4)
    x = cnn_aver(fk, x, drop=opt.drop - 0.1)  # 1
    # x = attention_layer(x, fk, r=3)


    x = cnn_aver(fk * 2, x, drop=opt.drop - 0.1)  # 2
    # x = attention_layer(x, fk*2, r=2)

    x = cnn_aver(fk * 4, x, drop=opt.drop - 0.1)  # 3
    # x = attention_layer(x, fk*4, r=2)
    x = Average()([x, res1])  # residual

    # res2 = MaxPool2D(padding='same', pool_size=(8, 8))(x)
    res2 = Conv2D(fk * 32, (8, 8), strides=(8, 8), padding='same')(x)  # residual 2*2 = 4
    # res2 = Dropout(opt.drop/1.5)(res2)
    # res2 = BatchNormalization()(res2)
    # res2 = LeakyReLU(alpha=0.3)(res2)

    x = cnn_aver(fk * 8, x, drop=opt.drop - 0.1)  # 4
    x = cnn_aver(fk * 16, x, drop=opt.drop)  # 5

    x = cnn_aver(fk * 32, x, padding='same', drop=opt.drop)  # 6

    x = Add()([x, res2])  # residual

    # x = cnn_aver(fk * 64, x, padding='valid', drop=opt.drop)  # 7

    x = Flatten()(x)
    x = Dropout(opt.drop)(x)

    # #
    # x = Dense(256)(x)
    # x = Dropout(opt.drop)(x)
    # x = BatchNormalization()(x)
    # x = Dense(512)(x)
    # x = Dropout(opt.drop + 0.2)(x)
    # # x = BatchNormalization()(x)
    # x = Dense(512)(x)
    # x = Dropout(opt.drop+0)(x)

    # x = LeakyReLU(alpha=1)(x)

    output = Dense(num_classes, activation='sigmoid')(x) #'softsign'

    # model = Model(inputs=[input10,input11,input12,input13, input20, input21, input22, input23], outputs=output)
    model = Model(inputs=input10, outputs=output)
    return model
