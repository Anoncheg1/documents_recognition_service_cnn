# -*- coding: utf-8 -*-
'''Group-Equivariant DenseNet for Keras.

# Reference
- [Rotation Equivariant CNNs for Digital Pathology](http://arxiv.org/abs/1806.03962).
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)

This implementation is based on the following reference code:
 - https://github.com/gpleiss/efficient_densenet_pytorch
 - https://github.com/liuzhuang13/DenseNet
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.applications.imagenet_utils import _obtain_input_shape
from tensorflow.python.keras.applications.imagenet_utils import preprocess_input as _preprocess_input
# from tensorflow.python.keras.engine.topology import get_source_inputs
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import AveragePooling2D
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import Conv2DTranspose
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import GlobalAveragePooling2D
from tensorflow.python.keras.layers import GlobalMaxPooling2D
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import MaxPooling2D
from tensorflow.python.keras.layers import Reshape
from tensorflow.python.keras.layers import UpSampling2D
from tensorflow.python.keras.layers import concatenate
from tensorflow.python.keras.layers import Cropping2D
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.python.keras.layers import ZeroPadding1D
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
# from tensorflow.python.keras_contrib.layers.convolutional import SubPixelUpscaling

def crop_to_fit(main, to_crop):
    cropped_skip = to_crop
    skip_size = K.int_shape(cropped_skip)[1]
    out_size = K.int_shape(main)[1]
    if skip_size > out_size:
        size_diff = (skip_size - out_size) // 2
        size_diff_odd = ((skip_size - out_size) // 2) + ((skip_size - out_size) % 2)
        cropped_skip = Cropping2D(((size_diff, size_diff_odd),) * 2)(cropped_skip)
    return cropped_skip


def __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=0.99, epsilon=1e-3, center=True, scale=True,
                beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros',
                moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                gamma_constraint=None, axis=-1, **kwargs):
    """Utility function to get batchnorm operation.

    # Arguments
        filters: filters in `Conv2D`
        kernel_size: height and width of the convolution kernel (tuple)
        strides: stride in 'Conv2D'
        padding: padding mode in `Conv2D`
        use_bias: bias mode in `Conv2D`
        kernel_initializer: initializer in `Conv2D`
        bias_initializer: initializer in `Conv2D`
        kernel_regularizer: regularizer in `Conv2D`
        use_gcnn: control use of gcnn
        conv_group: group determining gcnn operation
        depth_multiplier: Used to shrink the amount of parameters, used for fair Gconv/Conv comparison.
        name: name of the ops; will become `name + '_conv'`

    # Returns
        Convolution operation for `Conv2D`.
    """

    return BatchNormalization(
        axis=axis,
        momentum=momentum,
        epsilon=epsilon,
        center=center,
        scale=scale,
        beta_initializer=beta_initializer,
        gamma_initializer=gamma_initializer,
        moving_mean_initializer=moving_mean_initializer,
        moving_variance_initializer=moving_variance_initializer,
        beta_regularizer=beta_regularizer,
        gamma_regularizer=gamma_regularizer,
        beta_constraint=beta_constraint,
        gamma_constraint=gamma_constraint, **kwargs)


def __Conv2D(filters,
             kernel_size,
             strides=(1, 1),
             padding='valid',
             use_bias=True,
             kernel_initializer='he_normal',
             bias_initializer='zeros',
             kernel_regularizer=None,
             use_gcnn=None,
             conv_group=None,
             depth_multiplier=1,
             name=None):
    """Utility function to get conv operation, works with group to group
       convolution operations.

    # Arguments
        filters: filters in `Conv2D`
        kernel_size: height and width of the convolution kernel (tuple)
        strides: stride in 'Conv2D'
        padding: padding mode in `Conv2D`
        use_bias: bias mode in `Conv2D`
        kernel_initializer: initializer in `Conv2D`
        bias_initializer: initializer in `Conv2D`
        kernel_regularizer: regularizer in `Conv2D`
        use_gcnn: control use of gcnn
        conv_group: group determining gcnn operation
        depth_multiplier: Used to shrink the amount of parameters, used for fair Gconv/Conv comparison.
        name: name of the ops; will become `name + '_conv'`

    # Returns
        Convolution operation for `Conv2D`.
    """


    if depth_multiplier != 1:
        raise ValueError("Only use depth multiplier for gcnn networks.")

    return Conv2D(
        filters, kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        name=name)


def preprocess_input(x, data_format=None):
    """Preprocesses a tensor encoding a batch of images.

    # Arguments
        x: input Numpy tensor, 4D.
        data_format: data format of the image tensor.

    # Returns
        Preprocessed tensor.
    """
    x = _preprocess_input(x, data_format=data_format)
    x *= 0.017  # scale values
    return x


def name_or_none(prefix, name):
    return prefix + name if (prefix is not None and name is not None) else None


def __conv_block(ip, nb_filter, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn, bottleneck=False, dropout_rate=None,
                 weight_decay=1e-4, use_gcnn=None, conv_group=None, depth_multiplier=1, kernel_size=3,
                 block_prefix=None):
    '''
    Adds a convolution layer (with batch normalization and relu),
    and optionally a bottleneck layer.

    # Arguments
        ip: Input tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        bottleneck: if True, adds a bottleneck convolution block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        block_prefix: str, for unique layer naming

     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to stride.

    # Returns
        output tensor of block
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
        :param mc_dropout:
    '''
    with K.name_scope('ConvBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = ip
        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name=name_or_none(block_prefix, '_bn'))(x, training=mc_bn)
        x = Activation('relu')(x)

        if bottleneck:
            inter_channel = nb_filter * 4

            x = __Conv2D(inter_channel, (1, 1), kernel_initializer='he_normal', padding=padding, use_bias=False,
                         kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_bottleneck_conv2D'),
                         use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)(x)
            x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                            name=name_or_none(block_prefix, '_bottleneck_bn'))(x, training=mc_bn)
            x = Activation('relu')(x)

        x = __Conv2D(nb_filter, (kernel_size, kernel_size), kernel_initializer='he_normal', padding=padding,
                     use_bias=False,
                     name=name_or_none(block_prefix, '_conv2D'), use_gcnn=use_gcnn, conv_group=conv_group,
                     depth_multiplier=depth_multiplier)(x)
        if dropout_rate:
            if mc_dropout:
                x = Dropout(dropout_rate)(x, training=True)
            else:
                x = Dropout(dropout_rate)(x)

    return x


def __dense_block(x, nb_layers, nb_filter, padding, mc_dropout, bn_momentum, growth_rate, use_g_bn, mc_bn,
                  return_concat_list=False, block_prefix=None, bottleneck=False, dropout_rate=None, weight_decay=1e-4,
                  use_gcnn=None, conv_group=None, depth_multiplier=1, kernel_size=3, grow_nb_filters=True):
    '''
    Build a dense_block where the output of each conv_block is fed
    to subsequent ones

    # Arguments
        x: input keras tensor
        nb_layers: the number of conv_blocks to append to the model
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        growth_rate: growth rate of the dense block
        bottleneck: if True, adds a bottleneck convolution block to
            each conv_block
        dropout_rate: dropout rate
        weight_decay: weight decay factor
        grow_nb_filters: if True, allows number of filters to grow
        return_concat_list: set to True to return the list of
            feature maps along with the actual output
        block_prefix: str, for block unique naming

    # Return
        If return_concat_list is True, returns a list of the output
        keras tensor, the number of filters and a list of all the
        dense blocks added to the keras tensor

        If return_concat_list is False, returns a list of the output
        keras tensor and the number of filters
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
        :param mc_dropout:
    '''
    with K.name_scope('DenseBlock'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x_list = [x]

        for i in range(nb_layers):
            cb = __conv_block(x, growth_rate, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn, bottleneck=bottleneck,
                              dropout_rate=dropout_rate, weight_decay=weight_decay, use_gcnn=use_gcnn,
                              conv_group=conv_group, depth_multiplier=depth_multiplier, kernel_size=kernel_size,
                              block_prefix=name_or_none(block_prefix, '_%i' % i))
            x_list.append(cb)

            x = concatenate([crop_to_fit(cb, x), cb], axis=concat_axis)

            if grow_nb_filters:
                nb_filter += growth_rate

        if return_concat_list:
            return x, nb_filter, x_list
        else:
            return x, nb_filter


def __transition_block(ip, nb_filter, padding, bn_momentum, use_g_bn, mc_bn, block_prefix=None, compression=1.0,
                       weight_decay=1e-4, use_gcnn=None, conv_group=None, depth_multiplier=1):
    '''
    Adds a pointwise convolution layer (with batch normalization and relu),
    and an average pooling layer. The number of output convolution filters
    can be reduced by appropriately reducing the compression parameter.

    # Arguments
        ip: input keras tensor
        nb_filter: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        compression: calculated as 1 - reduction. Reduces the number
            of feature maps in the transition block.
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter * compression, rows / 2, cols / 2)`
        if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows / 2, cols / 2, nb_filter * compression)`
        if data_format='channels_last'.

    # Returns
        a keras tensor
        :param mc_bn:
        :param use_g_bn:
        :param bn_momentum:
        :param padding:
    '''
    with K.name_scope('Transition'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = ip
        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name=name_or_none(block_prefix, '_bn'))(x, training=mc_bn)
        x = Activation('relu')(x)
        x = __Conv2D(int(nb_filter * compression), (1, 1), kernel_initializer='he_normal', padding=padding,
                     use_bias=False, kernel_regularizer=l2(weight_decay), name=name_or_none(block_prefix, '_conv2D'),
                     use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)(x)
        x = AveragePooling2D((2, 2), strides=(2, 2))(x)

        return x


def __transition_up_block(ip, nb_filters, type='deconv', weight_decay=1E-4, block_prefix=None):
    '''Adds an upsampling block. Upsampling operation relies on the the type parameter.

    # Arguments
        ip: input keras tensor
        nb_filters: integer, the dimensionality of the output space
            (i.e. the number output of filters in the convolution)
        type: can be 'upsampling', 'subpixel', 'deconv'. Determines
            type of upsampling performed
        weight_decay: weight decay factor
        block_prefix: str, for block unique naming

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if data_format='channels_last'.

    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, rows * 2, cols * 2)` if data_format='channels_first'
        or 4D tensor with shape:
        `(samples, rows * 2, cols * 2, nb_filter)` if data_format='channels_last'.

    # Returns
        a keras tensor
    '''
    with K.name_scope('TransitionUp'):

        if type == 'upsampling':
            x = UpSampling2D(name=name_or_none(block_prefix, '_upsampling'))(ip)
        else:
            x = Conv2DTranspose(nb_filters, (3, 3), activation='relu', padding='valid', strides=(2, 2),
                                kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay),
                                name=name_or_none(block_prefix, '_conv2DT'))(ip)
        return x


def __create_dense_net(nb_classes, img_input, include_top, mc_dropout, padding, bn_momentum, use_g_bn, mc_bn,
                       nb_layers_per_block=-1, bottleneck=False, reduction=0.0, dropout_rate=None, weight_decay=1e-4,
                       subsample_initial_block=False, pooling=None, activation='softmax', depth=40, nb_dense_block=3,
                       growth_rate=12, use_gcnn=False, conv_group=None, depth_multiplier=1, kernel_size=3,
                       nb_filter=-1):
    ''' Build the DenseNet model

    # Arguments
        nb_classes: number of classes
        img_input: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        include_top: flag to include the final Dense layer
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end (generally = 3)
        growth_rate: number of filters to add per dense block
        nb_filter: initial number of filters. Default -1 indicates initial number of filters is 2 * growth_rate
        nb_layers_per_block: number of layers in each dense block.
                Can be a -1, positive integer or a list.
                If -1, calculates nb_layer_per_block from the depth of the network.
                If positive integer, a set number of layers per dense block.
                If list, nb_layer is used as provided. Note that list size must
                be (nb_dense_block + 1)
        bottleneck: add bottleneck blocks
        reduction: reduction factor of transition blocks. Note : reduction value is inverted to compute compression
        dropout_rate: dropout rate
        weight_decay: weight decay rate
        subsample_initial_block: Changes model type to suit different datasets.
            Should be set to True for ImageNet, and False for CIFAR datasets.
            When set to True, the initial convolution will be strided and
            adds a MaxPooling2D before the initial dense block.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        activation: Type of activation at the top layer. Can be one of 'softmax' or 'sigmoid'.
                Note that if sigmoid is used, classes must be 1.

    # Returns
        a keras tensor

    # Raises
        ValueError: in case of invalid argument for `reduction`
            or `nb_dense_block`
            :param mc_bn:
            :param use_g_bn:
            :param bn_momentum:
            :param padding:
            :param mc_dropout:
    '''
    with K.name_scope('DenseNet'):
        concat_axis = 1 if K.image_data_format() == 'channels_first' else -1

        if reduction != 0.0:
            if not (reduction <= 1.0 and reduction > 0.0):
                raise ValueError('`reduction` value must lie between 0.0 and 1.0')

        # layers in each dense block
        if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
            nb_layers = list(nb_layers_per_block)  # Convert tuple to list

            if len(nb_layers) != (nb_dense_block):
                raise ValueError('If `nb_dense_block` is a list, its length must match '
                                 'the number of layers provided by `nb_layers`.')

            final_nb_layer = nb_layers[-1]
            nb_layers = nb_layers[:-1]
        else:
            if nb_layers_per_block == -1:
                assert (depth - 4) % 3 == 0, 'Depth must be 3 N + 4 if nb_layers_per_block == -1'
                count = int((depth - 4) / 3)

                if bottleneck:
                    count = count // 2

                nb_layers = [count for _ in range(nb_dense_block)]
                final_nb_layer = count
            else:
                final_nb_layer = nb_layers_per_block
                nb_layers = [nb_layers_per_block] * nb_dense_block
        print('nb_layers computed:', nb_layers, final_nb_layer)

        # compute initial nb_filter if -1, else accept users initial nb_filter
        if nb_filter <= 0:
            nb_filter = 2 * growth_rate

        # compute compression factor
        compression = 1.0 - reduction

        # Initial convolution
        if subsample_initial_block:
            initial_kernel = (7, 7)
            initial_strides = (2, 2)
        else:
            initial_kernel = (1, 1)
            initial_strides = (1, 1)


        if depth_multiplier != 1:
            raise ValueError("Only use depth multiplier for gcnn networks.")

        x = Conv2D(nb_filter, initial_kernel, kernel_initializer='he_normal', padding=padding,
                       name='initial_conv2D',
                       strides=initial_strides, use_bias=False, kernel_regularizer=l2(weight_decay))(img_input)

        if subsample_initial_block:
            x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                            name='initial_bn')(x, training=mc_bn)
            x = Activation('relu')(x)
            x = MaxPooling2D((3, 3), strides=(2, 2), padding=padding)(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = __dense_block(x, nb_layers[block_idx], nb_filter, padding, mc_dropout, bn_momentum,
                                         growth_rate, use_g_bn, mc_bn, block_prefix='dense_%i' % block_idx,
                                         bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                         use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier,
                                         kernel_size=kernel_size)
            # add transition_block
            x = __transition_block(x, nb_filter, padding, bn_momentum, use_g_bn, mc_bn,
                                   block_prefix='tr_%i' % block_idx, compression=compression, weight_decay=weight_decay,
                                   use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier)
            nb_filter = int(nb_filter * compression)

        # The last dense_block does not have a transition_block
        x, nb_filter = __dense_block(x, final_nb_layer, nb_filter, padding, mc_dropout, bn_momentum, growth_rate,
                                     use_g_bn, mc_bn, block_prefix='dense_%i' % (nb_dense_block - 1),
                                     bottleneck=bottleneck, dropout_rate=dropout_rate, weight_decay=weight_decay,
                                     use_gcnn=use_gcnn, conv_group=conv_group, depth_multiplier=depth_multiplier,
                                     kernel_size=kernel_size)

        x = __BatchNorm(use_g_bn, conv_group, use_gcnn, momentum=bn_momentum, epsilon=1.1e-5, axis=concat_axis,
                        name='final_bn')(x, training=mc_bn)
        x = Activation('relu')(x)

        # if include_top:
        #     x = GlobalAveragePooling2D()(x)
        #     x = Dense(nb_classes, activation=activation)(x)
        # else:
        #     if pooling == 'avg':
        #         x = GlobalAveragePooling2D()(x)
        #     if pooling == 'max':
        #         x = GlobalMaxPooling2D()(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(nb_classes)(x)
        x = Activation(activation)(x)
        return x


from classes import siz

def get_densenet(num_classes: int, opt):
    input10 = Input(shape=(siz, siz, 3))
    output = __create_dense_net(num_classes, input10, include_top = False, mc_dropout=False, padding='same', bn_momentum= 0.9, use_g_bn=False, mc_bn=False,
                           growth_rate=12, reduction=0, dropout_rate=0.2, weight_decay=1e-4, nb_layers_per_block=2,
                           activation='softsign', conv_group=None, use_gcnn=False, nb_dense_block=8, pooling='avg', kernel_size=2)
    model = Model(inputs=[input10], outputs=output)
    return model