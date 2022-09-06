import keras
from cnn.sequence import CNNSequence
import os



if __name__ == '__main__':
    batch_size = 8

    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cnn_trained_model.h5'
    model_path = os.path.join(save_dir, model_name)

    seq = CNNSequence(batch_size, './test/')

    model = keras.models.load_model(model_path)
    scores = model.evaluate_generator(seq, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
