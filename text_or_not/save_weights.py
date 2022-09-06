# # own
# from predict_utils.classificator_cnn import Classifier, M
#
# orientation = Classifier(M.ORIENTATION_PASSPORT)
# import time
# time.sleep(100)


# from tensorflow_core.python import keras  # ignore error
# from tensorflow_core.python.keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import os
# use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# parent_path = os.path.join(os.getcwd(), os.pardir)
# save_dir = os.path.join(os.getcwd(), 'selected_models')  # must in current directory
# model_name = 'cnn_trained_model2019-09-12_123611.826453.h5'
# model_path = os.path.join(save_dir, model_name)
model_path = '/mnt/hit4/hit4user/PycharmProjects/cnn/text_or_not/saved_models/cnn_trained_model2020-10-07 10:06:36.703748.h5'
print(model_path)

model: Model = keras.models.load_model(model_path)
import time

with open("./model_weights/model_to_json.json", "w") as json_file:
    json_file.write(model.to_json(indent=4))

model.save_weights('./model_weights/weights.tf', save_format='tf')
print("ok")
time.sleep(100)