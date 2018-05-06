#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:29:00 2018

@author: fquinton
"""
import cv2 as cv
import numpy as np
import random
import computingFunctions as cpF

def isolate_color(img, color, INV=0, interval=20):

    #Convert BGR to HSV
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    hsv_color = cv.cvtColor(np.uint8([[color]]), cv.COLOR_BGR2HSV)
    lower= np.array([hsv_color[0][0][0]-interval,100,100])
    upper= np.array([hsv_color[0][0][0]+interval,255,255])


    # Threshold the HSV image to get only desired colors
    mask = cv.inRange(hsv, lower, upper)

    ## As a general rule of the thumb, you should try
    ## to repeat as little code as possible -
    ## The four first lines are the same for both
    ## cases, so we don't need to repeat them

    if INV:
        # Bitwise-AND inverted mask and original image
        res = cv.bitwise_and(img, img, mask=cv.bitwise_not(mask))
    else:
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(img, img, mask= mask)

    return(res)
    
def find_contours(img, slider_pos):
    imgray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,thresh = cv.threshold(imgray,slider_pos,255,0)
    image, contours, hierarchy = cv.findContours(thresh,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

def hierarchy_clean(img, isGivenContours=0, contours=None, hierarchy=None):
    if not isGivenContours:
        contours, hierarchy = find_contours(img)
    # This is more pythonic:
    res = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3]==-1]
    return(res)

class FitEllipse:

    def __init__(self, source_image, print_mode, slider_pos, color):
        self.source_image = source_image
        self.pm = print_mode
        # single line if statements should be reserved to very shor/simple cases

        if color is not None:
            self.source_image = isolate_color(self.source_image, color, 0)
        if print_mode:
            cv.createTrackbar("Threshold", "Result", slider_pos, 255, self.process_image)
        self.process_image(slider_pos)

    ## Usually, if a class as a single method, it might not be necessary
    ## to create a class - here you call the method at instanciation time,
    ## So I'm not sure you need a clas
    def process_image(self, slider_pos):
        """
        This function finds contours, draws them and their approximation by ellipses.
        """
        # Create the destination images
        if self.pm:
            image04 = self.source_image.copy()
        contours, hierarchy = find_contours(self.source_image, slider_pos)
        cont = hierarchy_clean(self.source_image, 1, contours, hierarchy)
        res = {}
        for c in cont:
            # Number of points must be more than or equal to 6 for cv.FitEllipse2
            if len(c) < 6:
                continue

            # Draw the current contour in gray
            if self.pm:
                cv.drawContours(image04, c,-1, [0,0,250],1)
            # Fits ellipse to current contour.
            ellipse = cv.fitEllipse(c)
            tuple_c = tuple([tuple(i[0]) for i in c])
            res[tuple_c] = ellipse
            # Draw ellipse in random color
            if self.pm :
                cv.ellipse(image04, ellipse,
                            [random.randrange(256),
                             random.randrange(256),
                            random.randrange(256)], 2)

        # Show image. HighGUI use.
        if self.pm:
            cv.imshow( "Result", image04 )
        return res # return is a statement, not a function

def fitEllipse(img, slider_pos=40, print_mode=False, color=None):
    if print_mode:
        cv.namedWindow("Source", 1)
        cv.namedWindow("Result", 1)
        cv.resizeWindow('Result', 600,600)
        cv.imshow("Source", img)
    fe = FitEllipse(img, print_mode, slider_pos, color).process_image(slider_pos)
    if print_mode:
        print("Press any key to exit")
        cv.waitKey(0)
        cv.destroyWindow("Source")
        cv.destroyWindow("Result")
    return fe

def membraneExtraction(img):
    img_red = isolate_color(img,[0, 0, 255])
    red_cont, red_hier = find_contours(img_red,0)
    
    outside = [contour for contour, hierarchy
              in zip(red_cont, red_hier[0])
              if hierarchy[3] == 0][0]
    
    
    inside = [contour for contour, hierarchy
              in zip(red_cont, red_hier[0])
              if hierarchy[3] == 2][0]
    
    (o_x,o_y), o_radius = cv.minEnclosingCircle(outside)
    (i_x,i_y), i_radius = cv.minEnclosingCircle(inside)
    return outside, inside, o_x, o_y, o_radius, i_x, i_y, i_radius


def nucleiDetection(img):
    img_blu = isolate_color(img,[255, 0, 0])
    ellipses = fitEllipse(img_blu,0,0)
    centers = [ellipses[i][0] for i in ellipses]

    return ellipses, centers

class dataFakeImg:
    
    def __init__(self, path):
        
        img = cv.imread(path)
        self.shape = img.shape
    
        
        """
        red pixels = membrane
        """
        membraneData = membraneExtraction(img)
        
        #saving useful data on the membrane
        outside = membraneData[0]
        inside = membraneData[1]
        o_x, o_y, self.rOut = membraneData[2:5]
        i_x,i_y, self.rIn = membraneData[5::]
        
        """
        blue pixels = nuclei
        """
        
        nucleiData = nucleiDetection(img)
        
        #saving useful data on the nuclei
        self.ellipses = nucleiData[0]
        centers = nucleiData[1]
        
        inside = (inside - np.ones(inside.shape)*(img.shape[0]/2.0,img.shape[1]/2.0))/float(img.shape[0])
        outside = (outside - np.ones(outside.shape)*(img.shape[0]/2.0,img.shape[1]/2.0))/float(img.shape[0])
        self.inside = np.squeeze(inside)
        self.outside = np.squeeze(outside)

        #for each cell defined by its center, compute its closest clockwise neighbor   
        neighbors = cpF.findNeighbor(centers, (o_x,o_y))
        #we need the list of cell nuclei to be clockwise ordered
        clockwiseCenters = [neighbors[centers[0]]]
        for i in centers[0:len(centers)-1] :
            tmp = clockwiseCenters[len(clockwiseCenters)-1]
            clockwiseCenters.append(neighbors[tmp])
        
        self.clockwiseCenters = np.array([[(clockwiseCenters[i][0]-self.shape[0]/2.0)/self.shape[0],(clockwiseCenters[i][1]-self.shape[1]/2.0)/self.shape[1]] for i in range(len(clockwiseCenters))])

class dataImage:
    
    def __init__(self, brigthfield_path, dapi_path):
        
        self.brigthfield = cv.imread(brigthfield_path,cv.IMREAD_GRAYSCALE)
        self.dapi = cv.imread(dapi_path)
        self.shape = self.dapi.shape
        
    def brightfield_processing(self):
        
        img = self.brigthfield.copy()
        
        ret,img = cv.threshold(img,28,255,0)

        img = cv.GaussianBlur(img,(9,9),0)
        
        img, contours, hierarchy = cv.findContours(img,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        contours_length = [(len(contours[i]),i) for i in range(len(contours))]
        contours_length.sort()
        membrane_ind = contours_length[len(contours_length)-2:]
        
        inside, outside = [contours[ind[1]] for ind in membrane_ind]       
        
        (self.o_x, self.o_y), self.rOut = cv.minEnclosingCircle(outside)
        (self.i_x, self.i_y), self.rIn = cv.minEnclosingCircle(inside) 
        
        inside = (inside - np.ones(inside.shape)*(img.shape[0]/2.0,img.shape[1]/2.0))/float(img.shape[0])
        outside = (outside - np.ones(outside.shape)*(img.shape[0]/2.0,img.shape[1]/2.0))/float(img.shape[0])
            
        self.inside = np.squeeze(inside)
        self.outside = np.squeeze(outside)
        
        
    def dapi_processing(self):  
        
        img = self.dapi.copy()
        
        ret,img = cv.threshold(img,8,255,0)
        
        #saving useful data on the nuclei
        
        self.ellipses = fitEllipse(img,0,0)
        
        centers = [self.ellipses[i][0] for i in self.ellipses]
        
        #for each cell defined by its center, compute its closest clockwise neighbor   
        neighbors = cpF.findNeighbor(centers, (self.o_x,self.o_y))
        
                #we need the list of cell nuclei to be clockwise ordered
        clockwiseCenters = [neighbors[centers[0]]]
        for i in centers[0:len(centers)-1] :
            tmp = clockwiseCenters[len(clockwiseCenters)-1]
            clockwiseCenters.append(neighbors[tmp])
        
        self.clockwiseCenters = np.array([[(clockwiseCenters[i][0]-self.shape[0]/2.0)/self.shape[0],(clockwiseCenters[i][1]-self.shape[1]/2.0)/self.shape[1]] for i in range(len(clockwiseCenters))])
