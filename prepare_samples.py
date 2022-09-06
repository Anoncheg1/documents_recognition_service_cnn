import os
import cv2 as cv
import numpy as np
import random
from shutil import copyfile
from classes import paths_other, paths_passport, paths_pts, siz
from shared_image_functions import crop, fix_angle, get_lines_c
from all_classes.count_files import count_samples_all
import re
import math
import imutils

# get from subdirs:
# find . -mindepth 2 -type f -print -exec mv {} . \;


class Counter:
    def __init__(self, amounts, multiplyer):
        self.lim: int = int(max(amounts) * multiplyer)
        print("Counter limit:", self.lim)

    def new_count(self, one_amount):
        self.c: int = 0
        self.r: int = math.ceil(self.lim / one_amount)

    def how_many_now(self) -> int:
        diff: int = 0
        r: int = self.r
        if (self.c + r) > self.lim:
            diff = self.c + r - self.lim

        self.c += r - diff  # update counter
        return r - diff


def datageg_keras(img, gen, directory, count=15):
    # gray img - (900, 900)
    # should be (samples, height, width, channels)
    #x = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # x = img.reshape((1,) + img.shape + (1,))
    # print(img.shape)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    x = img.reshape((1,) + img.shape)
    #print(x.shape)
    i = 0
    for _ in gen.flow(x, y=None, shuffle=False, save_to_dir=directory):
        i += 1
        if i >= count:  # save 20 images
            break  # otherwise the generator would loop indefinitely


def split(fp: str = './prep/', split_rate=0.2):
    def write_test_train(directory, files2):  # direcotry may be 'aaaa/bbb'

        try:
            os.makedirs('./train/' + directory + '/0/')
            os.mkdir('./train/' + directory + '/1/')
            os.mkdir('./train/' + directory + '/2/')
            os.mkdir('./train/' + directory + '/3/')
            os.makedirs('./test/' + directory + '/0/')
            os.mkdir('./test/' + directory + '/1/')
            os.mkdir('./test/' + directory + '/2/')
            os.mkdir('./test/' + directory + '/3/')
        except:
            print("ERROR cannot create", directory)
            exit(1)

        for ind, _ in enumerate(files2):  # 0,1,2,3
            read_dir = fp + directory + '/' + str(ind) + '/'
            write_dir_train = './train/' + directory + '/' + str(ind) + '/'
            write_dir_test = './test/' + directory + '/' + str(ind) + '/'
            # count split sizes
            ln = len(files2[ind])  # samples in directory
            test_size = int(ln * split_rate)
            train_size = ln - test_size

            random.shuffle(files2[ind])

            for i, file in enumerate(files2[ind]):

                fsource = read_dir + file
                if i < train_size:
                    ftarget = write_dir_train + file
                    copyfile(fsource, ftarget)
                else:
                    ftarget = write_dir_test + file
                    copyfile(fsource, ftarget)
    #  ---------------------------------------------
    for dir1 in os.listdir(fp):
        if os.path.isdir(fp + dir1):
            files = ([], [], [], [])
            for i in range(4):
                subd = dir1 + '/' + str(i) + '/'
                read_dir = fp + subd
                for file_name in os.listdir(read_dir):
                    if file_name.endswith(".png") or file_name.endswith(".jpg"):
                        files[i].append(file_name)

            write_test_train(dir1, files)


def split_passport(fr: str = './prep/', split_rate=0.2):
    """  from ./prep/ to ./train and ./test
    with shuffle inside class

    :param split_rate: test rate to all
    :return:
    """

    def write_test_train(dirs: dict, split_rate: float, passport: str):
        for cl in dirs.keys():  # classes

            try:
                os.makedirs('./train/' + passport + cl + '/0/')
                os.mkdir('./train/' + passport + cl + '/1/')
                os.mkdir('./train/' + passport + cl + '/2/')
                os.mkdir('./train/' + passport + cl + '/3/')
                os.makedirs('./test/' + passport + cl + '/0/')
                os.mkdir('./test/' + passport + cl + '/1/')
                os.mkdir('./test/' + passport + cl + '/2/')
                os.mkdir('./test/' + passport + cl + '/3/')
            except:
                pass

            for ind, _ in enumerate(dirs[cl]):  # 0,1,2,3

                read_dir = fr + passport + cl + '/' + str(ind) + '/'

                ln = len(dirs[cl][ind])  # samples in directory
                test_size = int(ln * split_rate)
                train_size = ln - test_size

                random.shuffle(dirs[cl][ind])

                for i, file in enumerate(dirs[cl][ind]):  # file = /filename.png

                    fsource = read_dir + file
                    if i < train_size:
                        ftarget = './train/' + passport + cl + '/' + str(ind) + '/' + file
                        copyfile(fsource, ftarget)
                    else:
                        ftarget = './test/' + passport + cl + '/' + str(ind) + '/' + file
                        copyfile(fsource, ftarget)


    # Split
    prep_passport = {}  # cl: [filepath]
    prep_other = {}  # cl: [filepath]

    # collect files for split
    for cl in paths_passport:
        prep_passport[cl] = ([], [], [], [])
        for i in range(4):
            read_dir = fr + 'passport/' + cl + '/' + str(i) + '/'
            for filename in os.listdir(read_dir):
                prep_passport[cl][i].append(filename)

    # for cl in paths_other:
    #     prep_other[cl] = ([], [], [], [])
    #     for i in range(4):
    #         read_dir = fr + cl + '/' + str(i) + '/'
    #         for filename in os.listdir(read_dir):
    #             prep_other[cl][i].append(filename)

    passport = 'passport/'
    write_test_train(prep_passport, split_rate, passport)

    # passport = ''
    # write_test_train(prep_other, split_rate, passport)


def count_samples(where: str, what: str) -> (dict, dict):
    """ max and min scans in samples folder
    used from classes import paths_other, paths_passport, paths_pts """

    def count_saplmes_dir(readp, directory, cdic) -> dict:
        for filename in os.listdir(readp):
            if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
                if directory in cdic:
                    cdic[directory] += 1
                else:
                    cdic[directory] = 1
        return cdic

    cdic = dict()
    full_passport_path = ['passport/' + vv for vv in paths_passport]
    full_pts_path = ['pts/' + vv for vv in paths_pts]
    full_path = dict()

    if what == 'passports':
        dirs = full_passport_path
    elif what == 'others':
        dirs = paths_other + full_pts_path
    else:
        dirs = full_passport_path + full_pts_path + paths_other

    for directory in dirs:
        full_path[directory] = list()  # sorted
        if where == './samples/' or where == './samples_validate/':
            for i in range(4):  # 0,1,2,3
                readp = where + directory + '/' + str(i)
                full_path[directory].append(readp)
                cdic = count_saplmes_dir(readp, directory, cdic)
        elif where == './prep_rot/':  # with bad
            readp = where + directory
            if os.path.exists(readp):
                full_path[directory].append(readp)
                cdic = count_saplmes_dir(readp, directory, cdic)
            readp = where + directory + '/bad'
            if os.path.exists(readp):
                full_path[directory].append(readp)
                cdic = count_saplmes_dir(readp, directory, cdic)
        elif where == './prep/':
            readp = where + directory
            full_path[directory].append(readp)
            cdic = count_saplmes_dir(readp, directory, cdic)


        # elif what == 'all':  # with subclasses
        #     readp = where + directory
        #     if os.path.exists(readp):
        #         cdic = count_saplmes_dir(readp, directory, cdic)
        #         full_path[directory].append(readp)
        # else:  # without bad
        #     readp = where + directory
        #     if os.path.exists(readp):
        #         full_path[directory].append(readp)
        #         cdic = count_saplmes_dir(readp, directory, cdic)

    if what == 'all':
        passports = 0
        ptss = 0
        delk = []
        for k, v in cdic.items():
            if 'passport/' in k:
                passports += v
                delk.append(k)
            if 'pts/' in k:
                ptss += v
                delk.append(k)

        for x in delk:
            del cdic[x]

        cdic['passport'] = passports
        cdic['pts'] = ptss

    return cdic, full_path


def samples_to_prep(count: dict, dirs: dict, crop=True):
    """Crop and Random rotate samples
     from ./samples to ./prep"""

    def sub_process(file_name):

        if file_name.endswith(".png") or file_name.endswith(".jpg"):  # photo
            pa = d + '/' + file_name
            print(pa)
            img = cv.imread(pa)  # numpy.ndarray - type

            if crop:
                img, _ = crop(img, rotate=True)
                if img is None:
                    print("can't crop", pa)
                    return
            else:
                img = img
                img = imutils.resize(img, width=int(siz*1.7))
                img = fix_angle(img, get_lines_c)
                img = cv.resize(img, (siz, siz))

            img_rotated = img
            for _ in range(4 - rotb):  # correct
                img_rotated = cv.transpose(img_rotated)
                img_rotated = cv.flip(img_rotated, flipCode=1)



            # irot = random.randint(0, 3)
            # rotafter = irot + rotb - 4 if (irot + rotb > 3) else irot + rotb

            # print("wtf", rotafter)
            # img2 = cv.resize(resized, (900, 900))
            # cv.imshow('image', img2)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window

            cv.imwrite(writep + '/' + file_name + '_' + str(rotb) + '.png', img_rotated)
            # datageg_keras(resized, mage_gen, write_dir[rotafter], count=1)




    # from tensorflow.python.keras.api._v2.keras.preprocessing.image import ImageDataGenerator
    prep = './prep/'
    # mage_gen = ImageDataGenerator(featurewise_center=True,
    #                               featurewise_std_normalization=True,
    #                               samplewise_std_normalization=0.2,
    #                               rotation_range=1,
    #                               width_shift_range=0.005,
    #                               height_shift_range=0.005,
    #                               shear_range=0.001,
    #                               zoom_range=[0.95, 1.00],
    #                               # horizontal_flip=True,
    #                               # vertical_flip=False,
    #                               # data_format='channels_last',
    #                               brightness_range=[0.9, 1.1],
    #                               zca_epsilon=1e-6,
    #                               channel_shift_range=0.1
    #                               )

    try:
        os.mkdir(prep)
    except:
        pass

    for directory in dirs.keys():
        # if directory != 'passport_and_vod': #or directory == 'vodit_udostav':
        #     continue
        writep = prep + directory  # prepared for split
        try:
            os.makedirs(writep)
        except:
            pass

        # write_dir = [writep + '/0', writep + '/1', writep + '/2', writep + '/3']
        # try:
        #     for wd in write_dir:
        #         os.makedirs(wd)
        # except:
        #     pass
        from multiprocessing import Process
        for rotb, d in enumerate(dirs[directory]):
            pool = []
            for i, filename in enumerate(os.listdir(d)):

                p = Process(target=sub_process, args=(filename,))
                pool.append(p)
                p.start()
                if i % 6 == 0:
                    for p in pool:
                        p.join()
            for p in pool:
                p.join()
    # with Pool(5) as p:
    #     p.map(sup_process, os.listdir(d))


    #img = np.moveaxis(img, -1, 0)  # shape (3, 150, 150)


def prep_to_fineprep_rot(count: dict, dirs: dict, multiplier: float = 1, siz: int = siz):  # prep rotated
    # from tensorflow_core.python.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    mage_gen = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  # samplewise_std_normalization=True,
                                  rotation_range=1,
                                  width_shift_range=0.005,
                                  height_shift_range=0.005,
                                  shear_range=0.003,
                                  # zoom_range=[0.95, 1.00],
                                  # horizontal_flip=True,
                                  # vertical_flip=False,
                                  # data_format='channels_last',
                                  brightness_range=[0.9, 1.1],
                                  # zca_epsilon=1e-6,
                                  channel_shift_range=0.1
                                  )
    fineprep = './fineprep/'
    try:
        os.mkdir(fineprep)
    except:
        pass

    counter = Counter(count.values(), multiplier)  # to equal prepared samples

    # create directories
    for k, v in dirs.items():
        writep = fineprep + k  # passport/
        write_dir = [writep + '/0', writep + '/1', writep + '/2', writep + '/3']
        try:
            for wd in write_dir:
                os.makedirs(wd)
        except:
            pass

        # count
        # count_ok = 0
        # count_bad = 0
        # for filename in os.listdir(readp):
        #     if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
        #         count_ok += 1
        # readp_bad = readp + '/' + 'bad'
        # if os.path.exists(readp_bad):
        #     for filename in os.listdir(readp_bad):
        #         if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
        #             count_bad += 1

        def gen(readp):
            for filename in os.listdir(readp):

                pa = readp + '/' + filename
                img = cv.imread(pa)  # numpy.ndarray - type

                hmn = counter.how_many_now()  # how many copies
                rot = random.randint(0, 3)  # initial rotation for first copy
                rots = [0, 1, 2, 3]*(hmn//4 + 2)  # 012301230123 sequence
                img_rotated = img
                for i in range(hmn):  # how many copies
                    r_now = rots[i+rot]
                    if r_now != 0:
                        for _ in range(r_now):  # change rot
                            img_rotated = cv.transpose(img_rotated)
                            img_rotated = cv.flip(img_rotated, flipCode=1)
                    img_rotated = cv.resize(img_rotated, (siz, siz))  # resize
                    datageg_keras(img_rotated, mage_gen, write_dir[rot], count=1)  # write 1

        counter.new_count(count[k])

        for readp in v:
            gen(readp)
            # if os.path.exists(readp_bad):
            #     gen(readp_bad)


def prep_to_fineprep(dirs: dict):  # deprecated
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    mage_gen = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  samplewise_std_normalization=0.2,
                                  rotation_range=1,
                                  width_shift_range=0.007,
                                  height_shift_range=0.007,
                                  shear_range=0.004,
                                  zoom_range=[0.95, 1.00],
                                  # horizontal_flip=True,
                                  # vertical_flip=False,
                                  # data_format='channels_last',
                                  brightness_range=[0.9, 1.1],
                                  zca_epsilon=1e-6,
                                  channel_shift_range=0.1
                                  )
    prep = './prep/'
    fineprep = './fineprep/'
    try:
        os.mkdir(fineprep)
    except:
        pass

    for directory in dirs.keys():
        readp = prep + directory
        writep = fineprep + directory
        write_dir = [writep + '/0', writep + '/1', writep + '/2', writep + '/3']
        try:
            for wd in write_dir:
                os.makedirs(wd)
        except:
            pass

        for filename in os.listdir(readp):
            # res = re.search('_[0-9]*.png$', filename)
            # if res:
            #     start = res.span()[0]
            #     filename[start+1:]
            pa = readp + '/' + filename
            img = cv.imread(pa)  # numpy.ndarray - type
            img_rotated = img
            rot = random.randint(0, 3)
            if rot != 0:
                for _ in range(rot):  # change rot
                    img_rotated = cv.transpose(img_rotated)
                    img_rotated = cv.flip(img_rotated, flipCode=1)
            datageg_keras(img_rotated, mage_gen, write_dir[rot], count=1)


def main():
    # 0)
    count, dirs = count_samples('../samples/', 'passports')
    # count, dirs = count_samples('./samples_validate/', 'all')
    print(count)
    exit()
    # print(dirs)
    # # 1)
    # samples_to_prep(count, dirs, crop=False)  # Rotate: from ./samples to ./prep
    # 2) filtering by hands
    # 3)
    count, dirs = count_samples_all('./prep/')
    # print(count)
    # print(dirs)
    # # 4)
    # prep_to_fineprep_rot(count, dirs, multiplier=0.3, siz=siz // 2 // 2)
    # 5)
    split('./fineprep/', 0)  # from ./fineprep/ to ./train and ./test


    import datetime
    print(datetime.datetime.now())
    # readp = './samples/' + 'passport/passport_main'
    # readp = './samples/' + 'photo'
    # readp = './samples/' + 'unknown'

    # readp = '/home/u/Desktop/passports_one_png/'
    # img = cv.imread(readp + '16--0.png')  # passport
    # img = cv.imread(readp + '92--0.png')  # passport
    # img = cv.imread(readp + '/0/' + '2019080116-2-0.png')  # passport

    # img = cv.imread(readp + '/0/' + '39-49-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/3/' + '201908019-2-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/3/' + '54-9-0.png')  # numpy.ndarray - type

    # img = cv.imread(readp + '/0/' + '3-11-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/0/' + '10-18-0.png')  # passport
    # img = cv.imread(readp + '/0/' + '40-50-0.png')  # passport
    # img = cv.imread(readp + '/0/' + '2019080126-2-5.png')  # passport



    # img = cv.imread(readp + '/0/' + '34-44-0.png')  # passport
    # img = cv.imread(readp + '/0/' + '2019080142-2-0.png')  # passport
    # img = cv.imread(readp + '/0/' + '41-51-0.png')  # passport



    # img = cv.imread(readp + '/0/' + '2019080110-2-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/0/' + '2019080138-2-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/0/' + '2019080149-2-0.png')  # passport



    # img = cv.imread(readp + '/0/' + '2019080116-2-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/0/' + '30-4-0.png')  # numpy.ndarray - type
    # img = cv.imread(readp + '/0/' + '2019080121-2-0.png')  # numpy.ndarray - type
    #
    #
    # p = './samples/photo/0/18-259-0.png'
    p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/59-191-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/92-224-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/8-16-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/photo/0/2-10-0.png'
    # p = '/mnt/hit4/hit4user/PycharmProjects/cnn/samples/pts/second/1/23-06-1.png'
    # img = cv.imread(p)
    # def aa(img):
    #     img = cv.resize(img, (60, 60))
    # aa(img)
    #
    # cv.imshow('image', img)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window

    # from shared_image_functions import rotate_img, get_lines_c
    # img = rotate_img(img, get_lines_c)
    # # # #
    # # # # # print(img)
    # # # #
    # # img, _ = crop_passport(img, rotate=True)
    # img2 = cv.resize(img, (900, 900))
    # cv.imshow('image', img2)  # show image in window
    # cv.waitKey(0)  # wait for any key indefinitely
    # cv.destroyAllWindows()  # close window

    # readp += '/3/'
    # for filename in os.listdir(readp):
    #     img = cv.imread(readp + filename)  # passport
    #     img = cropimg(img, document=True)
    #     img = cv.resize(img, (900, 900))
    #     cv.imwrite('./aaaa/' +'wwww'+filename, img)
    # img = rotate_image(img)

    #
    # img2 = cv.resize(img, (900, 900))


if __name__ == '__main__':
    main()

