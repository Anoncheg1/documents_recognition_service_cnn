import os
import cv2 as cv
import math
from tqdm import tqdm
import shutil
import random
from multiprocessing.pool import Pool
#
from classes import siz
from samples_counter import Counter

def extract_from_subdirs(path_source: str, target_path: str):
    """      find . -mindepth 2 -type f -print -exec mv {} . \;
    """
    try:
        os.mkdir(target_path)
    except:
        pass
    for folder_path, _, files in os.walk(path_source):
        for f in files:
            shutil.copy(os.path.join(folder_path, f), os.path.join(target_path, f))


def datageg_keras(img, gen, directory, count=15):
    """ save image count times with changes"""
    # gray img - (900, 900)
    # should be (samples, height, width, channels)
    #x = img.reshape((1,) + img.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    # x = img.reshape((1,) + img.shape + (1,))
    # print(img.shape)
    if count == 0:
        return
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    x = img.reshape((1,) + img.shape)
    #print(x.shape)
    for i, _ in enumerate(gen.flow(x, y=None, shuffle=False, save_to_dir=directory), start=1):
        if i >= count:
            break  # otherwise the generator would loop indefinitely
    # print(i+1, count)


def count_samples_dir(readp) -> int:
    """
    used by count_samples_subdits
    :param cdic: dictionary to add readp result
    :param readp:  path/folder
    """

    count = 0
    for filename in next(os.walk(readp))[2]:
        if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
            count += 1
    return count


def count_samples_subdits(where: str) -> (dict, dict):
    """ max and min scans in samples folder
    used from classes import paths_other, paths_passport, paths_pts """

    cdic = dict()

    subdirs = next(os.walk(where))[1]

    for subd in subdirs:
        count = count_samples_dir(os.path.join(where, subd))
        if subd in cdic:
            cdic[subd] += count
        else:
            cdic[subd] = count

    return cdic


def prep_to_fineprep(counts: dict, directory_path: str, medium_quantity: int,
                     size_x: int = siz, size_y: int = siz, fineprep='./fineprep/'):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    mage_gen = ImageDataGenerator(featurewise_center=True,
                                  featurewise_std_normalization=True,
                                  # samplewise_std_normalization=True,
                                  rotation_range=3,
                                  width_shift_range=0.005,
                                  height_shift_range=0.005,
                                  shear_range=0.003,
                                  zoom_range=[0.95, 1.00],
                                  # horizontal_flip=True,
                                  # vertical_flip=False,
                                  # data_format='channels_last',
                                  brightness_range=[0.9, 1.1],
                                  # zca_epsilon=1e-6,
                                  channel_shift_range=0.1
                                  )
    try:
        os.mkdir(fineprep)
    except:
        pass

    counter = Counter(medium_quantity)  # to equal prepared samples

    with Pool(processes=1) as process_pool:
        for directory, quantity in counts.items():
            print(directory)
            source_path = os.path.join(directory_path, directory)
            target_path = os.path.join(fineprep, directory)
            try:
                os.makedirs(target_path)
            except:
                pass

            counter.new_count(quantity)
            c = 1
            for filename in tqdm(next(os.walk(source_path))[2]):
                c += 1

                h = counter.how_many_now()
                if h == 0:
                    break
                pa = os.path.join(source_path, filename)
                img = cv.imread(pa)  # numpy.ndarray - type
                if img is None:
                    continue
                img_rotated = cv.resize(img, (size_x, size_y))  # resize

                # datageg_keras(img_rotated, mage_gen, target_path, count=h)
                process_pool.apply_async(datageg_keras, args=(img_rotated, mage_gen, target_path, h))
            print(c)


def split(fine_prep: str = './prep/', split_rate=0.2):
    def write_test_train(directory, files):  # direcotry may be 'aaaa/bbb'
        read_dir = os.path.join(fine_prep, directory)
        write_dir_train = './train/' + directory
        write_dir_test = './test/' + directory
        try:
            os.makedirs(write_dir_train)
            os.makedirs(write_dir_test)
        except:
            print("ERROR cannot create", directory)
            exit(1)


        # count split sizes
        ln = len(files)  # samples in directory
        test_size = int(ln * split_rate)
        train_size = ln - test_size

        random.shuffle(files)

        for i, file in enumerate(files):

            fsource = os.path.join(read_dir, file)
            if i < train_size:
                ftarget = os.path.join(write_dir_train, file)
                shutil.copyfile(fsource, ftarget)
            else:
                ftarget = os.path.join(write_dir_test, file)
                shutil.copyfile(fsource, ftarget)
    #  ---------------------------------------------
    for subdir in next(os.walk(fine_prep))[1]:  # dirs
        print(subdir)
        read_dir = os.path.join(fine_prep + subdir)
        files = []
        for file_name in next(os.walk(read_dir))[2]:  # files
            if file_name.lower().endswith(".png") or file_name.lower().endswith(".jpg"):
                files.append(file_name)

        write_test_train(subdir, files)


def main():
    # 2)
    counts_not_text = count_samples_subdits('./samples/not_text')
    counts_text = count_samples_dir('./samples/text')

    print(len(counts_not_text), counts_not_text)  # {'vodit_udostav': 348, 'photo': 598, 'passport_and_vod': 255, 'unknown': 531, 'passport': 4357, 'pts': 1190}
    print(counts_text)  # 363
    # # 1)
    try:
        shutil.rmtree('./fineprep2/', ignore_errors=True)
        shutil.rmtree('./fineprep/', ignore_errors=True)
        shutil.rmtree('./train/', ignore_errors=True)
        shutil.rmtree('./test/', ignore_errors=True)
    except:
        pass
    # # test
    prep_to_fineprep({'text': counts_text}, './samples/', medium_quantity=2400,
                     size_x=round(siz // 2 // 2), size_y=siz // 2)  # one subdir
    # not_text
    prep_to_fineprep(counts_not_text, directory_path='./samples/not_text', medium_quantity=2400 // 4,
                     size_x=round(siz // 2 // 2), size_y=siz // 2, fineprep='./fineprep2/')  # with many subdirs
    #
    extract_from_subdirs('./fineprep2/', './fineprep/not_text')

    # 5)
    split('./fineprep/', 0.001)  # from ./fineprep/ to ./train and ./test
    # 6)
    shutil.rmtree('/dev/shm/train', ignore_errors=True)
    shutil.rmtree('/dev/shm/test', ignore_errors=True)
    shutil.copytree('./train', '/dev/shm/train')
    shutil.copytree('./test', '/dev/shm/test')

    import datetime
    print(datetime.datetime.now())

if __name__ == '__main__':
    main()

