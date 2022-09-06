from tensorflow_core.python import keras
import numpy as np
import cv2 as cv
import os
import random
#my
from classes import paths_passport, paths_other
# from shared_image_functions import rotate_img, get_lines_h, get_lines_c, crop_passport

PASSPORT_PAGE = 'passport_page'
O_P_METHOD = 'orientation_passport'
PASSPORT_MAIN = 'passport_main'
ALL_CLASSES = 'all_classes'


class CNNSequence(keras.utils.Sequence):

    def __init__(self, batch_size: int, mdir: str, opt):
        """
        :param batch_size:
        :param mdir:  './train/' or './test/'
        """

        x = []
        y = []

        y_rotate = keras.utils.to_categorical(range(4))
        other_len = len(paths_other)
        y_passport_other = keras.utils.to_categorical(range(other_len+1))  # passports and others
        passport_len = len(paths_passport)
        y_passport_type = keras.utils.to_categorical(range(passport_len))  # passports
        y_passport_main = keras.utils.to_categorical(range(2))  # passports
        # print("wtf",y_passport_other)

        for ic, directory in enumerate(paths_passport):  # passport
            pa = mdir + 'passport/' + directory
            for ir in (0, 1, 2, 3):
                pa2 = pa + '/' + str(ir) + '/'
                for filename in os.listdir(pa2):
                    # rn = random.randint(0, len(paths_passport))  # select 1/7 of passport from all groups(7)
                    # if rn != 0:  # passports and others
                    #     continue

                    if opt.model == PASSPORT_PAGE:
                        x.append([pa2 + filename, directory, ir])
                        y.append(y_passport_type[ic])  # passport type
                    elif opt.model == O_P_METHOD:
                        x.append([pa2 + filename, directory, ir])
                        y.append(y_rotate[ir])  # rotation passport
                    elif opt.model == PASSPORT_MAIN:
                        if directory == 'passport_main':
                            x.append([pa2 + filename, directory, ir])
                            y.append(y_passport_main[0])  # passport main or not
                        else:
                            rn = random.randint(0, (len(paths_passport)-3))  # select 1/((7-1)/2) of passport from all groups(7)
                            if rn != 0:  # passports and others
                                continue
                            x.append([pa2 + filename, directory, ir])
                            y.append(y_passport_main[1])
                        # y.append(y_passport_other[other_len])  # passports and others

        # if opt.model == O_P_METHOD:
        #     for ic, directory in enumerate(paths_other):  # other
        #         if directory == 'photo':
        #             continue
        #         pa = mdir + directory
        #         for ir in (0, 1, 2, 3):
        #             pa2 = pa + '/' + str(ir) + '/'
        #             for filename in os.listdir(pa2):
        #                 x.append([pa2 + filename, directory, ir])
        #                 # y.append(y_passport_other[ic])
        #                 y.append(y_rotate[ir])  # rotation other

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
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        """
        :param idx:
        :return: np.array(x), np.array(y) with size of batch
        """

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        x10 = list()  # batch
        # x11 = list()  # batch
        # x12 = list()  # batch
        # x13 = list()  # batch
        for file_name, directory, rotation in batch_x:
            im = cv.imread(file_name)  # RGB # 'std::out_of_range'  what():  basic_string::substr: __pos (which is 140) > this->size() (which is 0)
            if im is None:
                print("Sample image not found in sequence")
            # im, _ = rotate_image(im, get_lines_c)  # TODO: remove after new prepare

            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            if self.opt.model == PASSPORT_PAGE \
                    or self.opt.model == PASSPORT_MAIN:
                for _ in range(4-rotation):
                    timg = cv.transpose(im)
                    im = cv.flip(timg, flipCode=1)

            # im = (255 - im)
            # b1 = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
            # # _, b2 = cv.threshold(b1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            # # b3 = cv.Canny(im, 100, 255, apertureSize=3)
            # #
            # im = np.stack([b1,b1,b1], axis=2)
            #

            im = (255 - im)  # range[0,1]
            im = im / 255.0
            # im = im / 128.0  # range[0,2] bad
            # im = (2 - im)
            # im = (255 - im)  # range[0,2]
            # im = im / 128.0
            im = im.reshape(im.shape + (1,))  # channels
            x10.append(im)
            # xr1 = im
            #
            #
            # x10.append(xr1)
            # timg = cv.transpose(xr2)
            # xr3 = cv.flip(timg, flipCode=1)
            # timg = cv.transpose(xr3)
            # xr4 = cv.flip(timg, flipCode=1)
            # np.concatenate([xr1, xr2, xr3, xr4], axis=0)

            # xr1 = xr1.reshape(xr1.shape + (1,))  # channels
            # print(idx)


            # xr2 = xr2.reshape(xr2.shape + (1,))  # channels
            # x11.append(xr2)
            # xr3 = xr3.reshape(xr3.shape + (1,))  # channels
            # x12.append(xr3)
            # xr4 = xr4.reshape(xr4.shape + (1,))  # channels
            # x13.append(xr4)

            # im = im + [rotation]
            #
            # im = np.array([im, [1]])
            # x2 = r.reshape(r.shape + (1,))

            # # print(x2.shape)
            # im = np.array(im)
            # print(im.shape)


            # s = list(im.shape)
            # s[0] = s[0]*4
            # MASK


            # x2p = np.zeros(im.shape)
            # x2p = [x2p, x2p, x2p, x2p]
            # if rand == 0:  # 1/4 random signalize
            #     if rotation == 0:
            #         # x2p = np.concatenate([x2p + 1, x2p, x2p, x2p], axis=0)
            #         x2p[0] = x2p[0] + 1
            #     elif rotation == 1:
            #         # x2p = np.concatenate([x2p, x2p + 1, x2p, x2p], axis=0)
            #         x2p[1] = x2p[1] + 1
            #     elif rotation == 2:
            #         # x2p = np.concatenate([x2p, x2p, x2p + 1, x2p], axis=0)
            #         x2p[2] = x2p[2] + 1
            #     elif rotation == 3:
            #         # x2p = np.concatenate([x2p, x2p, x2p, x2p + 1], axis=0)
            #         x2p[3] = x2p[3] + 1
            #
            # for i, _ in enumerate(x2p):
            #     x2p[i] = x2p[i].reshape(x2p[i].shape + (1,))
            #
            # x20.append(x2p[0])
            # x21.append(x2p[1])
            # x22.append(x2p[2])
            # x23.append(x2p[3])


            # print(x2p.shape)
            # x2p = np.concatenate([x2p[0], x2p[1], x2p[2], x2p[3]], axis=0)
            # x2p = x2p.reshape(x2p.shape + (1,))
            # print(x2p.shape)
            # x2.append(x2p)


        # return [x10, x11, x12, x13, x20, x21, x22, x23], np.array(batch_y)
        # return [x10, x11, x12, x13], np.array(batch_y)
        return np.array(x10), np.array(batch_y)
