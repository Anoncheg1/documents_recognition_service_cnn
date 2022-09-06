import tensorflow as tf
from tensorflow.python.keras.api._v2.keras import backend as K
from classes import siz

def rotate_input(xinp):
    x1 = xinp[0]
    rotate_mask = xinp[1]
    # print(xinp[1].shape)
    # transformation = tf.contrib.image.angles_to_projective_transforms(90, siz, siz)
    # x2 = tf.contrib.image.transform(x1, transformation, output_shape=(siz, siz))
    # x3 = tf.contrib.image.transform(x2, transformation, output_shape=(siz, siz))
    # x4 = tf.contrib.image.transform(x3, transformation, output_shape=(siz, siz))
    #
    # ret = K.concatenate([x1,x2,x3,x4], axis=1) #, x_2, x_3
    # shape = list(xinp[0].shape)
    # shape[1] = 3
    z = K.zeros(xinp[0].shape)
    ret = K.concatenate([xinp[0],z, xinp[1],xinp[2],xinp[3]], axis=1)
    # ret = x1 + rotate_mask
    return ret #(? , 199, 199, 8)


def rotate_input_output_shape(input_shape):
    shape = list(input_shape)

    shape[1] *= 4
    return tuple(shape)
