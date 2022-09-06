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
model_path = '/mnt/hit4/hit4user/PycharmProjects/cnn/signature_or_not/saved_models/cnn_trained_model2020-12-16 08:27:08.517080.h5'
print(model_path)

model: Model = keras.models.load_model(model_path)


from sequence_simple import CNNSequence_Simple
from signature_or_not.options import options_set
opt = options_set()
direc = ['.', '/dev/shm']
d = direc[1]  # CHOOSE! local 0 or memory 1
train_seq = CNNSequence_Simple(opt.batchSize, d + '/train/', opt)
# print(train_seq.__getitem__(1))


def my_auc(labels, predictions):
    auc_metric = keras.metrics.AUC(name="my_auc")
    auc_metric.update_state(y_true=labels, y_pred=predictions['logistic'])
    return {'auc': auc_metric}


history = model.evaluate(x=train_seq)

print(history)
