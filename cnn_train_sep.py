# Эксперимент заморозки то верха то низа и тренировки их отдельно
# from __future__ import print_function
import os

import tensorflow as tf
from tensorflow.python.keras import backend as K
import logging
import datetime
import math

# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sequence import CNNSequence
import classes
import options
#from models.resnet_one_atten_model import get_resnet_model  # passport type
# from models.densenet_model import get_densenet  # passport type
# from models.highway_model import get_model  # passport type
from models.resnet_one_atten_model import get_model
# from models.resnet_gcnn_model import get_model  # rotation-invariant

# from cnn.capsule_model import get_resnet_model
import signal
import sys

stop_signal = False


def signal_handler(signal, frame):
    global stop_signal
    print('You pressed Ctrl+C!')
    stop_signal = True
    # sys.exit(0)


def save_model(model):
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)


# export PYTHONPATH="${PYTHONPATH}:~/PycharmProjects/rec2/"
if __name__ == '__main__':

    signal.signal(signal.SIGINT, signal_handler)  # Stop if KeyboardInterrupt

    # import time
    # time.sleep(60*60*6) # 6H
    # tfConfig = tf.ConfigProto(allow_soft_placement=True)
    # tfConfig.gpu_options.allow_growth = True
    # session = tf.Session(config=tfConfig)

    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

        # disable logger
        logging.getLogger('tensorflow').disabled = True

        opt = options.set()

        # -------------------
        save_dir = os.path.join(os.getcwd(), 'saved_models')

        d = datetime.datetime.now()
        model_name = "cnn_trained_model{}.h5".format(d)
        print("modelname = ", model_name)

        model_path = os.path.join(save_dir, model_name)

        direc = ['.', '/dev/shm']
        d = direc[1]  # CHOOSE! local 0 or memory 1
        train_seq = CNNSequence(opt.batchSize, d + '/train/', opt)
        test_seq = CNNSequence(opt.batchSize, d + '/test/', opt)

        # model = get_resnet_model(len(cnn.classes.paths_other)+1, opt)  # passports and others
        # model = get_resnet_model(len(classes.paths_passport), opt)  # passport type
        # model = get_densenet(len(classes.paths_passport), opt)  # passport type
        if opt.model == 'passport_type':
            model = get_model(len(classes.paths_passport), opt)  # passport type
        elif opt.model == 'orientation_passport':
            model = get_model(4, opt)  # orientation
        elif opt.model == 'passport_main':
            model = get_model(2, opt)  # orientation
        # model = get_resnet_model(4, opt)  # rotation
        # from cnn.residual_attention_model import AttentionResNet92
        # model = AttentionResNet92()

        # initiate RMSprop optimizer
        # opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


        # Let's prep the model using RMSprop
        model.compile(loss=opt.loss, #"squared_hinge",  #'mean_squared_error',  #  "categorical_crossentropy", #"squared_hinge",  # "categorical_crossentropy",  # loss='categorical_crossentropy',
                      optimizer=opt.optim,
                      metrics=['accuracy'])

        for i, layer in enumerate(model.layers):
            print(i)
            print("{}: {}".format(layer, layer.trainable))
            # layer.trainable = True

        print(model.summary())
        print("model=", opt.model)
        print('opt.optim=', opt.optimizer, "lr=", opt.lr, " d=", opt.decay)
        print("drop=", opt.drop, "momentum=", opt.momentum)
        print("net=", opt.net)

        # dense1 = 103  # highway
        # dense2 = 106
        # dense1 = 66  # gcnn
        # dense2 = 69
        dense1 = 142#150  # res_one_att
        dense2 = 146#154
        shuffle = True
        n = 2300 // opt.epochs

        acc = []
        loss = []
        val_acc = []
        val_loss = []
        for ii in range(opt.epochs):
            s1 = n + (n // 3) * ii
            s2 = n + (n // 2) * ii
            # print("epoch:", ii)
            # K.set_value(model.optimizer.lr, opt.lr / (math.exp(ii / opt.decay)))  # decay learning rate
            # print(K.get_value(model.optimizer.lr))
        # if i == 0:
        #     shuffle = False

            for i, layer in enumerate(model.layers):
                layer.trainable = True
            for layer in model.layers[dense1:dense2]:  # 101- enable with activation
                layer.trainable = False
            # print(model.summary())  # print model structure
            # Fit the model on the batches generated by datagen.flow().
            print("wtf")
            history = model.fit_generator(
                train_seq,
                steps_per_epoch=s1,  # only this first batches
                # epochs=1,
                epochs=(ii)*2,
                # validation_data=test_seq,
                # validation_freq=1,
                shuffle=shuffle,
                # use_multiprocessing=True,
                # workers=3
                initial_epoch=(ii)*2 - 1
            )
            for i, layer in enumerate(model.layers):
                layer.trainable = False
            for layer in model.layers[dense1:dense2]:  # 101- enable
                layer.trainable = True

            if ii > opt.epochs / 5:
                print("wtf2")
                history = model.fit_generator(
                    train_seq,
                    steps_per_epoch=s2,  # total images
                    epochs=(ii)*2 + 1,
                    validation_data=test_seq,
                    validation_freq=1,
                    shuffle=shuffle,
                    # use_multiprocessing=True,
                    # workers=3
                    initial_epoch=(ii)*2
                )
                acc.append(history.history['accuracy'])
                loss.append(history.history['loss'])
                val_acc.append(history.history['val_accuracy'])
                val_loss.append(history.history['val_loss'])
            else:
                print("wtf3")
                history = model.fit_generator(
                    train_seq,
                    steps_per_epoch=s2,  # total images
                    epochs=(ii)*2 + 1,
                    # validation_data=test_seq,
                    # validation_freq=0,
                    shuffle=shuffle,
                    # use_multiprocessing=True,
                    # workers=3
                    initial_epoch=(ii)*2
                )
                acc.append(history.history['accuracy'])
                loss.append(history.history['loss'])
            if stop_signal:
                break



            if True:
                from matplotlib import pyplot as plt

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



        # history = model.fit_generator(
        #     train_seq,
        #     # steps_per_epoch=2,  # only this first batches
        #     epochs=opt.epochs,
        #     validation_data=test_seq,
        #     validation_freq=1,
        #     shuffle=shuffle
        #     # use_multiprocessing=True,
        #     # workers=3
        # )


        # save history
        import json

        # with open('file.json', 'w') as f:
        #     json.dump(history.history, f)
        # with open('file.json', 'r', encoding='utf-8') as f:
        #     n = json.loads(f.read())






        # --------- plot ----------------

        # print(history.history.keys())
        # from matplotlib import pyplot as plt
        #
        # plt.figure(1)
        #
        # # summarize history for accuracy
        #
        # plt.subplot(211)
        # plt.plot(history.history['accuracy'])
        # plt.plot(history.history['val_accuracy'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        #
        # # summarize history for loss
        #
        # plt.subplot(212)
        # plt.plot(history.history['loss'])
        # plt.plot(history.history['val_loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
