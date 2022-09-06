from tensorflow import keras
import numpy as np
import cv2 as cv
import os
import random
#my
from classes import paths_passport, paths_other, paths_pts, all_classes, siz
# from shared_image_functions import rotate_img, get_lines_h, get_lines_c, crop_passport


siz = siz // 2 // 2  # SMALL SIZE!


class CNNSequence_all(keras.utils.Sequence):

    def __init__(self, batch_size: int, mdir: str, opt):
        """
        :param batch_size:
        :param mdir:  './train/' or './test/'
        """
        self.test = False
        if mdir == './test/':
            self.test = True

        x = []
        y = []

        all_len = len(all_classes) - 1  # passport_and_vod - not separate class
        y_all = keras.utils.to_categorical(range(all_len))
        # add class as sum of 0 and 3 classes - passport and vod_ud
        y_all = np.append(y_all, [y_all[0] + y_all[3]], axis=0)


        for cl in all_classes:
            pa = mdir + cl
            for ir in (0, 1, 2, 3):
                pa2 = pa + '/' + str(ir) + '/'
                for filename in os.listdir(pa2):
                    x.append([pa2 + filename, cl, ir])
                    # im = cv.imread(pa2 + filename)
                    # im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
                    # im = cv.resize(im, (siz, siz))
                    # x.append([im, cl, ir])
                    ind = all_classes.index(cl)
                    y.append(y_all[ind])

        self.batch_size = batch_size

        self.x = x
        self.y = y

        self.opt = opt

        # --- Divide epoch # i guess it was bad idea
        # div_epoch = 3  # 1 or 2 now
        # div_len = len(x) // div_epoch
        #
        # self.x = []
        # self.y = []
        # x2 = []
        # y2 = []
        #
        # for i, z in enumerate(zip(x, y)):
        #     if i < div_len:
        #         self.x.append(z[0])
        #         self.y.append(z[1])
        #     else:
        #         x2.append(z[0])
        #         y2.append(z[1])
        #
        # self.x = self.x + x2
        # self.y = self.y + y2

    def __len__(self):
        # if self.test:
        #     z = list(zip(self.x, self.y))
        #     z = random.shuffle(z)
        #     self.x, self.y = zip(*z)
        #     return int((np.ceil(len(self.x)//2) / float(self.batch_size)))
        # else:
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: np.array(x), np.array(y) with size of batch
        """

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x10 = list()
        for file_name, directory, rotation in batch_x:
            im = cv.imread(file_name, cv.IMREAD_GRAYSCALE)  # RGB # 'std::out_of_range'  what():  basic_string::substr: __pos (which is 140) > this->size() (which is 0)
            if im is None:
                print("Sample image not found in sequence")

            # im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            # im = cv.resize(im, (siz, siz))
            # cv.imshow('image', im)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window

            im = (255 - im)  # range[0,1]
            im = im / 255.0

            im = im.reshape(im.shape + (1,))  # channels
            x10.append(im)
            # print(directory, batch_y)

        return np.array(x10), np.array(batch_y)
