# import cv2 as cv
# import imutils
# import numpy as np
#
# img = cv.imread('/mnt/hit4/hit4user/PycharmProjects/cnn/samples/passport/passport_main/0/1-242-0.png')
# img2 = imutils.resize(img, 900)
# img2 = np.rot90(img2)
# cv.imshow('Result', img2)
# cv.waitKey()



import cv2 as cv
import numpy as np
p = '/home/u2/Pictures/bottom_passn1.png'
img = cv.imread(p)
# img2 = np.rot90(img)
img = cv.transpose(img)
img = cv.flip(img, flipCode=0)  # counterclockwise
img = cv.transpose(img)
img = cv.flip(img, flipCode=0)  # counterclockwise
img = cv.transpose(img)
img = cv.flip(img, flipCode=0)  # counterclockwise
cv.imwrite('/home/u2/Pictures/bottom_passn2.png', img)
# cv.imshow('Result', img)
# cv.waitKey()