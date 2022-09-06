import os
import cv2 as cv
import numpy as np
import random
from shutil import copyfile
from all_classes.count_files import count_samples_all

import math
from tqdm import tqdm


# get from subdirs:
# find . -mindepth 2 -type f -print -exec mv {} . \;


def count_samples(where: str, what: str) -> (dict, dict):
    def count_saplmes_dir(readp, directory, cdic) -> dict:
        for filename in os.listdir(readp):
            if filename.endswith(".png") or filename.endswith(".jpg"):  # photo
                if directory in cdic:
                    cdic[directory] += 1
                else:
                    cdic[directory] = 1
        return cdic

    cdic: dict = {what: 0}
    cdic = count_saplmes_dir(where, what, cdic)
    full_path: dict = {what: []}
    for filename in os.listdir(where):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            full_path[what].append(os.path.join(where, filename))

    return cdic, full_path


def samples_to_prep(count: dict, dirs: dict, crop=True):
    """Crop and Random rotate samples
     from ./samples to ./prep"""

    prep_sign = './prep/sign'
    prep_nosign = './prep/nosign'
    try:
        os.mkdir('./prep/')
        os.mkdir(prep_sign)
        os.mkdir(prep_nosign)
    except:
        pass

    for i, filename in tqdm(enumerate(dirs['signs'])):
        for _ in range(2):
            # read doct without text that will be cropped
            if i == 0 or random.randint(0, 7) == 5:
                d = random.sample(dirs['docs'], 1)[0]
                doc = cv.imread(d, cv.IMREAD_GRAYSCALE)
                height, width = doc.shape
            # # subdoc
            # h = 300
            # w = 300
            # # random subimage
            # alt = random.randint(-280, +50)
            # y = random.randint(0, height - h - alt)
            # x = random.randint(0, width - w - alt)
            # subdoc = doc[y:y + h + alt, x:x + w + alt]
            # subdoc = cv.resize(subdoc, dsize=(h, w))
            # sign
            # sign = cv.imread(filename, cv.IMREAD_COLOR)  # BGR
            # sign = cv.resize(sign, dsize=(h, w))
            # sign_gray = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)
            #
            # sign_gray = 255 - sign_gray
            # ret, mask = cv.threshold(sign_gray, 60, 255, cv.THRESH_TOZERO)
            # # mask = 255 - mask
            # mask_inv = cv.bitwise_not(mask)
            # # mask_inv = 255 - mask_inv
            # # Now black-out the area of logo in ROI
            # subdoc_bg = cv.bitwise_and(subdoc, subdoc, mask=mask_inv)
            # # Take only region of logo from logo image.
            # sign_fg = cv.bitwise_and(sign, sign, mask=mask)
            # # Put logo in ROI and modify the main image
            # print(subdoc_bg.shape, sign_fg.shape)
            # # sign_fg[:,:,0]*=0
            # # sign_fg[:,:,1]*=20
            # # sign_fg[:,:,1]*=0
            # # print(sign_fg.shape)
            # dst = cv.bitwise_and(subdoc_bg, sign_fg, mask=mask)
            # print(dst.shape)
            #
            # cv.imshow('image', mask)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window
            #
            # cv.imshow('image', dst)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window



            # sign
            sign = cv.imread(filename, cv.IMREAD_GRAYSCALE)
            sign = 255 - sign
            ret, sign = cv.threshold(sign, 60, 255, cv.THRESH_TOZERO)

            # RANDOM SHIFT SING TODO: random resize
            M = np.float32([[(random.random() + 0.2) * 1.7, 0, random.randint(-70, 70)],
                            [0, (random.random() + 0.2) * 1.7, random.randint(-70, 70)]])
            sign = cv.warpAffine(sign, M, sign.shape)

            sign = 255 - sign
            # random brightness
            ret, sign = cv.threshold(sign, random.randint(60, 200), 255, cv.THRESH_TOZERO)
            h = 300
            w = 300
            # random subimage
            alt = random.randint(-280, +50)
            y = random.randint(0, height - h - alt)
            x = random.randint(0, width - w - alt)
            subdoc = doc[y:y + h + alt, x:x + w + alt]
            subdoc = cv.resize(subdoc, dsize=(h, w))
            sign = cv.resize(sign, dsize=(h, w))

            sign_b = sign

            sign = cv.bitwise_and(subdoc, subdoc, mask=sign)

            # blue colour and save in BGR
            if random.randint(1,2) == 2:
                sign_b = cv.cvtColor(sign_b, cv.COLOR_GRAY2BGR)
                sign = cv.cvtColor(sign, cv.COLOR_GRAY2BGR)

                sign_b = cv.bitwise_not(sign_b)
                sign_b[:, :, 1] //= 2
                sign_b[:, :, 0] //= random.randint(2,30)
                sign_b = cv.bitwise_not(sign_b)

            sign = cv.bitwise_and(sign, sign_b)

            # cv.imshow('image', sign)  # show image in window
            # cv.waitKey(0)  # wait for any key indefinitely
            # cv.destroyAllWindows()  # close window

            cv.imwrite(os.path.join(prep_sign, str(i) + '.png'), sign)
            cv.imwrite(os.path.join(prep_nosign, str(i) + '.png'), subdoc)





def main():
    # 0)
    count, files = count_samples('./samples/signs', 'signs')
    p = '/home/u2/h4/PycharmProjects/cnn/text_or_not/samples/text/not/without_writings'
    count_docs, files_docs = count_samples(p, 'docs')
    # print(count, files)  # {'signs': 1320} {'signs': ['./samples/signs/original_28_6.png', './samples/signs/ori
    # print(count_docs, files_docs)
    count.update(count_docs)
    files.update(files_docs)
    # print(count, files)

    # 1)
    samples_to_prep(count, files, crop=False)  # Rotate: from ./samples to ./prep
    exit()
    # 2) filtering by hands
    # 3)
    count, dirs = count_samples_all('./prep/')
    print(count)
    print(dirs)
    # # 4)
    # prep_to_fineprep_rot(count, dirs, multiplier=0.3, siz=siz // 2 // 2)
    # 5)


if __name__ == '__main__':
    main()
