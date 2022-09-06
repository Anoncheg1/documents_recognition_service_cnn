# from __future__ import print_function
import os

import tensorflow as tf
import logging
from tensorflow import keras
from tensorflow.python.keras.engine.training import Model
import datetime
import math

from matplotlib import pyplot as plt
import warnings
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import classes
# import options


# from cnn.capsule_model import get_resnet_model
import signal
import sys

# import pandas as pd

stop_signal = False


def signal_handler(signal, frame):
    global stop_signal
    print('You pressed Ctrl+C!')
    stop_signal = True
    # sys.exit(0)


def save_model(model: Model):
    import datetime
    from tensorflow.python.keras.saving.save import save_model
    # Save model and weights
    d = datetime.datetime.now()
    model_name = "cnn_trained_model{}.h5".format(d)
    print("modelname = ", model_name)
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_path = os.path.join(save_dir, model_name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    # model.save(model_path)

    save_model(
        model,
        model_path,
        overwrite=True,
        include_optimizer=True,
        save_format="h5",
        signatures=None,
        options=None
    )

    print('Saved trained model at %s ' % model_path)


# export PYTHONPATH="${PYTHONPATH}:~/PycharmProjects/rec2/"
def main(options_set: callable, cnnseq: keras.utils.Sequence):
    signal.signal(signal.SIGINT, signal_handler)  # Stop if KeyboardInterrupt

    # import time
    # time.sleep(60*60*6) # 6H
    # tfConfig = tf.ConfigProto(allow_soft_placement=True)
    # tfConfig.gpu_options.allow_growth = True
    # session = tf.Session(config=tfConfig)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        # try:
        #     # Currently, memory growth needs to be the same across GPUs
        #     for gpu in gpus:
        #         tf.config.experimental.set_memory_growth(gpu, True)
        #     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        # except RuntimeError as e:
        #     # Memory growth must be set before GPUs have been initialized
        #     print(e)
        # Restrict TensorFlow to only allocate 5GB of memory on the first GPU
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

        # disable logger
        logging.getLogger('tensorflow').disabled = True

        opt = options_set()

        # -------------------
        if opt.net == 'mobilenet':
            from models.mobilenet import get_model
        elif opt.net == 'resnet_one_atten_model':
            from models.resnet_one_atten_model import get_model
        elif opt.net == 'resnet_one_atten_model_d3':
            from models.resnet_one_atten_model_d3_p4 import get_model
        elif opt.net == 'inception_one_atten_model':
            from models.inception_one_atten_model import get_model
        elif opt.net == 'hnet_model':
            from signature_or_not.hnet_model import get_model
        elif opt.net == 'keras':
            from tensorflow.keras.applications.mobilenet import MobileNet
            from tensorflow.keras.layers import Input, Dense, Flatten
            inputs = Input((opt.size_y, opt.size_x, 3))
            base_model: Model = MobileNet(input_shape=(opt.size_y, opt.size_x, 3),
                                      include_top=False, weights=None,
                                      classes=2, classifier_activation=None,
                                      pooling='max')(inputs)
            output = Dense(1, activation='sigmoid')(base_model)
            model: Model = Model(inputs, output)

        # model: Model = get_model(1, opt)

        # -------------------
        direc = ['.', '/dev/shm']
        d = direc[1]  # CHOOSE! local 0 or memory 1

        train_seq = cnnseq(opt.batchSize, d + '/train/', opt)
        test_seq = cnnseq(opt.batchSize, d + '/test/', opt)

        model.compile(loss=opt.loss,
                      optimizer=opt.optim,
                      metrics=['accuracy'])

        print(model.summary())
        print("model=", opt.model)
        print('opt.optim=', opt.optimizer, "lr=", opt.lr, " d=", opt.decay)
        print("drop=", opt.drop, "momentum=", opt.momentum)
        print("net=", opt.net)

        acc = []
        loss = []
        val_acc = []
        val_loss = []
        for i in range(opt.epochs):
            if stop_signal:
                break

            history = model.fit(
                x=train_seq,  # generator
                # steps_per_epoch=200,  # batches in epoch
                epochs=i + 1,
                validation_data=test_seq,
                validation_freq=1,
                # validation_steps=50,
                # use_multiprocessing=True,
                # workers=3
                initial_epoch=i
            )

            acc.append(history.history['accuracy'])
            loss.append(history.history['loss'])
            val_acc.append(history.history['val_accuracy'])
            val_loss.append(history.history['val_loss'])

            # Early stopping
            estop = acc
            if len(estop) > (opt.estop - 1) and estop[-opt.estop] > estop[-1]:
                print("EARLY STOPPING!!")
                break

            if True:
                warnings.simplefilter("ignore")
                plt.figure(1)

                plt.ion()
                plt.show()

                # summarize history for accuracy

                plt.subplot(211)
                plt.plot(acc)
                plt.plot(val_acc)
                plt.title('model accuracy')
                plt.ylabel('accuracy')
                # plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')

                # summarize history for loss

                plt.subplot(212)
                plt.plot(loss)
                plt.plot(val_loss)
                plt.title('model loss')
                plt.ylabel('loss')
                # plt.xlabel('epoch')
                plt.legend(['train', 'test'], loc='upper left')

                plt.pause(0.001)  # stay opened
                # plt.close()  # prepare for next cycle

        save_model(model)

        import time
        time.sleep(60 * 60 * 24)  # do not close plt

        # save history
        import json

        # with open('file.json', 'w') as f:
        #     json.dump(history.history, f)
        # with open('file.json', 'r', encoding='utf-8') as f:
        #     n = json.loads(f.read())

        # --------- plot ----------------

        # print(history.history)
        # from matplotlib import pyplot as plt
        #
        # plt.figure(1)
        #
        # # summarize history for accuracy
        #
        # plt.subplot(211)
        # plt.plot(acc)
        # plt.plot(val_acc)
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        #
        # # summarize history for loss
        #
        # plt.subplot(212)
        # plt.plot(loss)
        # plt.plot(val_loss)
        # plt.title('model loss')
        # plt.ylabel('loss')
        # # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()


if __name__ == '__main__':
    from signature_or_not.options import options_set
    from sequence_simple import CNNSequence_Simple

    main(options_set, CNNSequence_Simple)
