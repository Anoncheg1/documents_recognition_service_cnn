import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras  # ignore error
from tensorflow_core.python.keras.engine.training import Model
import os
from tensorflow_core.python.keras.layers.core import Flatten
from tensorflow_core.python.keras.layers.convolutional import Conv2D
from tensorflow_core.python.keras.layers.convolutional import Conv1D
from tensorflow_core.python.keras.engine.input_layer import Input
from tensorflow_core.python.keras.layers.normalization_v2 import BatchNormalization

def save_model(model):
    import datetime
    # Save model and weights
    d = datetime.datetime.now()
    model_name = "cnn_trained_model{}.h5".format(d)
    print("modelname = ", model_name)
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


num_imgs = 50000

img_size = 16
min_object_size = 1
max_object_size = 4
num_objects = 2

bboxes = np.zeros((num_imgs, num_objects, 4))  # x, y, w, h for each box
imgs = np.zeros((num_imgs, img_size, img_size))  # set background to 0

for i_img in range(num_imgs):
    for i_object in range(num_objects):
        w, h = np.random.randint(min_object_size, max_object_size, size=2)
        x = np.random.randint(0, img_size - w)
        y = np.random.randint(0, img_size - h)
        imgs[i_img, x:x + w, y:y + h] = 1.  # set rectangle to 1
        bboxes[i_img, i_object] = [x+w/2, y+h/2, w, h]  # save coordinats

print(imgs.shape, bboxes.shape)

i = 0
# plt.imshow(imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
# plt.show()
# for bbox in bboxes[i]:
#     plt.gca().add_patch(matplotlib.patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], ec='r', fc='none'))



# Reshape and normalize the image data to mean 0.0624359375 and std 1.
X = (imgs.reshape(num_imgs, -1) - np.mean(imgs)) / np.std(imgs)
# X = (imgs - np.mean(imgs)) / np.std(imgs)
print(X[0])
# print(np.std(imgs))
# print(np.mean(imgs))
print(X.shape, np.mean(X), np.std(X))
# exit(0)
# Normalize x, y, w, h by img_size, so that all values are between 0 and 1.
# Important: Do not shift to negative values (e.g. by setting to mean 0), because the IOU calculation needs positive w and h.
y = bboxes.reshape(num_imgs, -1) / img_size   # 0-8 -> 0-1
print(y.shape, np.mean(y), np.std(y))

# Split training and test.
i = int(0.8 * num_imgs)
train_X = X[:i]
test_X = X[i:]
train_y = y[:i]
test_y = y[i:]
test_imgs = imgs[i:]
test_bboxes = bboxes[i:]

if __name__ == '__main__':
    # Build the model.
    from tensorflow_core.python.keras.models import Sequential
    from tensorflow_core.python.keras.layers.core import Dense, Activation, Dropout
    # from tensorflow_core.python.keras.optimizers import SGD
    model: Model = Sequential([
            # Conv1D(40, 40, 1),
            # Input(shape=(X.shape[:1])),
            Dense(400, input_dim=X.shape[-1]),
            # BatchNormalization(),
            Dropout(0.2),
            Activation('relu'),
            Dense(400),
            Dropout(0.2),
            BatchNormalization(),
            Activation('relu'),
            Dense(400),
            Dropout(0.2),
            # BatchNormalization(),
            Activation('relu'),
            Dense(y.shape[-1]),
            # Activation('relu')
        ])
    opt = keras.optimizers.Adam(lr=5e-4)


    # flipped train
    flipped_train_y = np.array(train_y)

    num_epochs = 30
    flipped = np.zeros((len(flipped_train_y), num_epochs))
    ious_epoch = np.zeros((len(flipped_train_y), num_epochs))
    dists_epoch = np.zeros((len(flipped_train_y), num_epochs))
    mses_epoch = np.zeros((len(flipped_train_y), num_epochs))

    for epoch in range(num_epochs):
        print('Epoch', epoch)
        model.fit(train_X, flipped_train_y, epochs=1, validation_data=(test_X, test_y), verbose=2)
        pred_y = model.predict(train_X)

    # model.compile(opt, 'mean_squared_logarithmic_error', metrics=['accuracy'])
    # # model.summary()
    #
    # model.fit(train_X, train_y, epochs=30, validation_data=(test_X, test_y), verbose=2, validation_freq=10)
    #
    # save_model(model)
