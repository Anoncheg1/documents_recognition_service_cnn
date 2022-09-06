from tensorflow import keras
from tensorflow.keras.models import Model
from models import resnet_one_atten_model
import os
# use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# parent_path = os.path.join(os.getcwd(), os.pardir)
model_path = '/mnt/hit4/hit4user/PycharmProjects/cnn/signature_or_not/saved_models/cnn_trained_model2020-12-23 05:08:14.400416.h5'
print(model_path)

model: Model = keras.models.load_model(model_path)
import time

name = 'cnn_hw_or_not2020-12-23 05:08:14.400416'
os.mkdir(name)


with open("./"+name+"/model_to_json.json", "w") as json_file:
    json_file.write(model.to_json(indent=4))

model.save_weights('./'+name+'/weights.tf', save_format='tf')
print("ok")
time.sleep(1)
