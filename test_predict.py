# Predict bounding boxes on the test images.
from tensorflow import keras  # ignore error
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from test import test_X, img_size, num_objects, test_imgs, test_bboxes, test_y
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_trained_model2019-09-05 17:58:53.783551.h5'
model_path = os.path.join(save_dir, model_name)
# print(model_path)

model = keras.models.load_model(model_path)

pred_y = model.predict(test_X)
print(pred_y.shape)
pred_bboxes = pred_y * img_size
pred_bboxes = pred_bboxes.reshape(len(pred_bboxes), num_objects, -1)
print(pred_bboxes.shape)



def IOU(bbox1, bbox2):
    '''Calculate overlap between two bounding boxes [x, y, w, h] as the area of intersection over the area of unity'''
    x1, y1, w1, h1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
    x2, y2, w2, h2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]

    w_I = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_I = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_I <= 0 or h_I <= 0:  # no overlap
        return 0.
    I = w_I * h_I

    U = w1 * h1 + w2 * h2 - I

    return I / U


# Show a few images and predicted bounding boxes from the test dataset.
plt.figure(figsize=(12, 3))
for i_subplot in range(1, 5):
    plt.subplot(1, 4, i_subplot)
    i = np.random.randint(len(test_imgs))
    plt.imshow(test_imgs[i].T, cmap='Greys', interpolation='none', origin='lower', extent=[0, img_size, 0, img_size])
    for pred_bbox, exp_bbox in zip(pred_bboxes[i], test_bboxes[i]):
        plt.gca().add_patch(matplotlib.patches.Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], ec='r', fc='none'))
        plt.annotate('IOU: {:.2f}'.format(IOU(pred_bbox, exp_bbox)), (pred_bbox[0], pred_bbox[1]+pred_bbox[3]+0.2), color='r')

# plt.show()
# Calculate the mean IOU (overlap) between the predicted and expected bounding boxes on the test dataset.
summed_IOU = 0.
for pred_bbox, test_bbox in zip(pred_bboxes.reshape(-1, 4), test_bboxes.reshape(-1, 4)):
    summed_IOU += IOU(pred_bbox, test_bbox)
mean_IOU = summed_IOU / len(pred_bboxes)
print(mean_IOU)


print(model.predict(np.array([test_X[0]])))
print(test_y[0])
