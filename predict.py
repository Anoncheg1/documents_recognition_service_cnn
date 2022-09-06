from tensorflow import keras
from sequence_simple import CNNSequence_Simple
import os


if __name__ == '__main__':
    batch_size = 3

    # save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_path = '/mnt/hit4/hit4user/PycharmProjects/cnn/text_or_not/saved_models/cnn_trained_model2020-09-10 09:26:34.553480.h5'
    # model_path = os.path.join(save_dir, model_name)

    # use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # parent_path = os.path.join(os.getcwd(), os.pardir)

    seq = CNNSequence_Simple(batch_size, '/dev/shm/train/')

    model = keras.models.load_model(model_path)

    scores = model.evaluate_generator(seq, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])