#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 16:05:05 2018

@author: fquinton
"""

import numpy as np
import cv2 as cv
#import imageProcessing as iP
import matplotlib.pyplot as plt

path = '../images/Images/r01c01f03p55-ch1sk1fk1fl1.tiff'

img = cv.imread(path,cv.IMREAD_GRAYSCALE)

copy = img.copy()

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
plt.imshow(copy, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

ret,img = cv.threshold(img,8,255,0)

img, contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

contours = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3]==-1]

copy = cv.cvtColor(copy, cv.COLOR_GRAY2BGR)

cv.drawContours(copy, contours,-1, [0,0,250],5)

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
plt.imshow(copy, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

path = '../images/Images/r01c01f03p55-ch3sk1fk1fl1.tiff'


img = cv.imread(path,cv.IMREAD_GRAYSCALE)

copy = img.copy()

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

ret,img = cv.threshold(img,28,255,0)

img = cv.GaussianBlur(img,(9,9),0)


#img[img>28] = 255

img, contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

contours_length = [(len(contours[i]),i) for i in range(len(contours))]
contours_length.sort()
membrane_ind = contours_length[len(contours_length)-2:]


contours = [contours[ind[1]] for ind in membrane_ind]

copy = cv.cvtColor(copy, cv.COLOR_GRAY2BGR)

cv.drawContours(copy, contours,-1, [0,0,250],5)

fig, ax = plt.subplots()
fig.set_size_inches(8, 8)
plt.imshow(copy, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()