from tensorflow import keras
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


class CNNSequence_Simple(keras.utils.Sequence):

    def __init__(self, batch_size: int, mdir: str, opt=None):
        """
        :param batch_size:
        :param mdir:  './train/' or './test/'
        """

        x = []
        y = []

        subdires = next(os.walk(mdir))[1]
        self.variants = len(subdires)
        if self.variants == 2:
            self.variants = 1
        categories = keras.utils.to_categorical(range(self. variants))  # passports

        for i, subdir in enumerate(subdires):  # dirs
            for file_name in next(os.walk(os.path.join(mdir, subdir)))[2]:
                x.append(os.path.join(mdir, subdir, file_name))
                if self.variants == 1:
                    y.append(i)
                else:
                    y.append(categories[i])

        z = list(zip(x, y))
        random.shuffle(z)
        x, y = list(zip(*z))

        self.batch_size = batch_size

        self.x = x
        self.y = y

        self.opt = opt

    def __len__(self):
        return round(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        """
        :param idx:
        :return: np.array(x), np.array(y) with size of batch
        """

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        bx = list()  # batch
        # x11 = list()  # batch
        # x12 = list()  # batch
        # x13 = list()  # batch
        for file_path in batch_x:
            im = cv.imread(file_path)  # RGB # 'std::out_of_range'  what():  basic_string::substr: __pos (which is 140) > this->size() (which is 0)
            # im = cv.resize(im, (144, 144))
            if im is None:
                print("Sample image not found in sequence", file_path)
            # im, _ = rotate_image(im, get_lines_c)  # TODO: remove after new prepare

            # im.shape = (144, 144, 3) ALWAYS
            if self.opt.channels == 1:
                im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

            # im = np.rot90(im)
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
            if self.opt.channels == 1:
                im = im.reshape(im.shape + (1,))  # channels
            # print(im.shape)
            bx.append(im)
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
        # print(np.array(batch_y).shape)
        # print(np.array(batch_y).reshape((-1, 1, 1, self.variants)).shape)
        # print(np.array(bx).shape)
        return np.array(bx), np.array(batch_y)  #np.array(batch_y).reshape((-1,  self.variants))


if __name__ == '__main__':
    # test
    s = CNNSequence_Simple(8, '/dev/shm/test')
    print("len", s.__len__())
    print("get_item", s.__getitem__(11)[1].shape, s.__getitem__(11)[1])
