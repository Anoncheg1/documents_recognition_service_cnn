import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.api._v2.keras.layers import MaxPool2D
from tensorflow.python.keras.api._v2.keras.layers import BatchNormalization, Dense, Dropout, Activation, Flatten, \
    Conv2D, MaxPooling2D, LocallyConnected2D, Lambda, Input
from tensorflow.python.keras.api._v2.keras import backend as K
from tensorflow.python.keras.api._v2.keras.regularizers import l2
from tensorflow.python.keras.api._v2.keras.layers import AveragePooling2D
from tensorflow.python.keras.api._v2.keras.models import Model

from residual_blocks import residual_block
from residual_blocks import attention_block

from cnn.rotate_methods import rotate_input, rotate_input_output_shape

from classes import siz, num_classes

num_classes = len(paths)


def AttentionResNet92(shape=(siz, siz, 1), n_channels=14, n_classes=num_classes,
                      dropout=0.06, regularization=0.004):
    """
    Attention-92 ResNet
    https://arxiv.org/abs/1704.06904
    """
    regularizer = l2(regularization)

    input_ = Input(shape=shape)
    x = Lambda(rotate_input, output_shape=rotate_input_output_shape, input_shape=(siz, siz, 1))(input_)
    # x = Conv2D(n_channels, (7, 7), strides=(2, 2), padding='same')(x)  # 112x112
    # x = Conv2D(n_channels, (2, 2), strides=(1, 1), padding='same')(x)  # 112x112
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)  # 56x56
    # print(x)
    x = residual_block(x, output_channels=n_channels * 4)  # 56x56
    x = attention_block(x, encoder_depth=3)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = residual_block(x, output_channels=n_channels * 8, stride=2)  # 28x28
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7
    x = attention_block(x, encoder_depth=2)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    x = residual_block(x, output_channels=n_channels * 16, stride=2)  # 14x14
    # x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    # x = attention_block(x, encoder_depth=1)  # bottleneck 7x7
    # x = attention_block(x, encoder_depth=1)  # bottleneck 7x7

    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32, stride=2)  # 7x7
    x = residual_block(x, output_channels=n_channels * 32)
    # x = residual_block(x, output_channels=n_channels * 32)

    # pool_size = (x.get_shape()[1].value, x.get_shape()[2].value)
    # x = AveragePooling2D(pool_size=pool_size, strides=(1, 1))(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1))(x)
    x = Flatten()(x)
    if dropout:
        x = Dropout(dropout)(x)

    # x = Dense(200, kernel_regularizer=regularizer, activation=None)(x)
    # x = Dropout(dropout)(x)
    # x = BatchNormalization()(x)
    # x = keras.layers.LeakyReLU(alpha=0.3)(x)

    output = Dense(n_classes, kernel_regularizer=regularizer, activation='softsign')(x) #softmax

    model = Model(input_, output)
    return model