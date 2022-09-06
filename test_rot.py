import numpy as np
import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import h5py
# from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras.losses import squared_hinge

# losses.mean_squared_error
# import keras
# from keras.datasets import mnist
from tensorflow_core.python.keras.models import Sequential
from tensorflow_core.python.keras.engine.training import Model
from tensorflow_core.python.keras.losses import MeanSquaredError
from tensorflow_core.python.keras.layers.core import Dense, Dropout, Flatten
from tensorflow_core.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow_core.python.keras.layers.core import Dropout
# from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Activation
from tensorflow_core.python.keras.layers.normalization_v2 import BatchNormalization
# from keras.optimizers import SGD

MASK_VALUE = -2
n = 25  # # datapoints
n_tasks = 19  # tasks / # binary classes
input_dim = 2048  # vector size

# generate random X vectors and random
# Y labels (binary labels [0,1] or -1 for missing value
x = np.random.rand(n, input_dim).astype(np.float32)  # 0-1
# print(x)
x_test = np.random.rand(5, input_dim)
y_orig = np.random.randint(3, size=(n, n_tasks)).astype(np.float32)-1  # -1-2 shape = n, n_tasks
y = y_orig.copy()
y[:,n_tasks-10:] = -2
print(y)

def build_masked_loss(loss_function, mask_value=MASK_VALUE):
    """Builds a loss function that masks based on targets
    Args:
        loss_function: The loss function to mask
        mask_value: The value to mask in the targets
    Returns:
        function: a loss function that acts like loss_function with masked inputs
    """

    def masked_loss_function(y_true, y_pred):
        dtype = K.floatx()
        mask = K.cast(K.not_equal(y_true, mask_value), dtype)  # -2 -> 0 other -> 1
        a = y_true * mask
        b = y_pred * mask
        return loss_function(a, b)

    return masked_loss_function


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    mask = K.cast(K.not_equal(y_true, MASK_VALUE), dtype)  # -2 -> 0 other -> 1
    total = K.cast(K.sum(mask), dtype)
    y = y_pred * mask
    correct = K.sum(K.cast(K.equal(y_true, K.round(y)), dtype))  # everything not unmasked
    return correct / total


# create model
model: Model = Sequential()
model.add(Dense(1000, input_dim=input_dim))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.7))
model.add(Dense(512))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.7))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.7))
model.add(Dense(n_tasks, activation='softsign'))
model.compile(loss=build_masked_loss(squared_hinge), optimizer='adam', metrics=[masked_accuracy]) #K.binary_crossentropy

model.fit(x, y,
          epochs=10)

y = y_orig.copy()
y[:,:n_tasks-10] = -2
print(y)
model.fit(x, y,
          epochs=10)

# model.fit(x, y_orig,
#           epochs=10)
model.evaluate(x, y_orig)

