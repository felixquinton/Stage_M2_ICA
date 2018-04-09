#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 15:51:48 2018

@author: fquinton
"""

import numpy as np
import cv2
import random
from scipy.spatial import Delaunay
from scipy.spatial import Voronoi



# The comment bellow should be the function's docstring
# style guide for numpy-related project docstring
# can be found here:
# https://numpydoc.readthedocs.io/en/latest/format.html

# I don't tend to use opencv for display, as it is a bit cumbersome
# (e.g. you have to comform to their data type)
# matplotlib is more generic

"""
Isoler les pixels d'une couleur donnée

img = image cv2
color = couleur à isoler. triplet [x,y,z] code couleur BGR. La recherche est effectuée dans
un interval autour de cette couleur contrôlé par la parametre interval
INV = booléen. si 0 la fonction renvoie une image avec seulement les pixels de la couleur color
si 1 la fonction renvoie une image sans les pixels de la couleur color.
interval = int. largeur de l'interval.optional. défaut = 10.
"""

# The INV argument should have a more explicit name
# like inverted, and default to False, not 0, as a boolean
# is expected
def isolate_color(img, color, INV=0, interval=20):

    #Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
    lower= np.array([hsv_color[0][0][0]-interval,100,100])
    upper= np.array([hsv_color[0][0][0]+interval,255,255])


    # Threshold the HSV image to get only desired colors
    mask = cv2.inRange(hsv, lower, upper)

    ## As a general rule of the thumb, you should try
    ## to repeat as little code as possible -
    ## The four first lines are the same for both
    ## cases, so we don't need to repeat them

    if INV:
        # Bitwise-AND inverted mask and original image
        res = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    else:
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img, img, mask= mask)

    return(res)

# This should be a docstring
"""
Trouve les contours d'une image en BGR.
Retourne la liste des contours et le tableau de hierarchie
"""

def find_contours(img, slider_pos):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,slider_pos,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours, hierarchy

# This should be a docstring
"""
Retourne les contours de hierarchie 0 pour une image BGR
"""

def hierarchy_clean(img, isGivenContours=0, contours=None, hierarchy=None):
    if not isGivenContours:
        contours, hierarchy = find_contours(img)
    # This is more pythonic:
    res = [contour for i, contour in enumerate(contours) if hierarchy[0][i][3]==-1]
    return(res)


# This should be a docstring
"""
Retourne le squelette d'une liste de contours. prend en entrée l'image et la lsite de contours
retourne
"""

def skeleton(img, contours):
    pass


# This should be a docstring
"""
Fonction/classe fitEllipse qui associe à chaque contour une ellispe min
Affichage avec barre graduée pour régler le treshold
print_mode = 1 -> image GUI. =0 -> pas de print juste un dico associant
contours et ellipse.
slider_pos = int <255. indique la borne inf de pixels a garder dans le threshold.
color = couleur à conserver. liste BGR ou si pas de couleur = -1 (default)
"""
class FitEllipse:

    def __init__(self, source_image, print_mode, slider_pos, color):
        self.source_image = source_image
        self.pm = print_mode
        # single line if statements should be reserved to very shor/simple cases

        if color is not None:
            self.source_image = isolate_color(self.source_image, color, 0)
        if print_mode:
            cv2.createTrackbar("Threshold", "Result", slider_pos, 255, self.process_image)
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
                cv2.drawContours(image04, c,-1, [0,0,250],1)
            # Fits ellipse to current contour.
            ellipse = cv2.fitEllipse(c)
            tuple_c = tuple([tuple(i[0]) for i in c])
            res[tuple_c] = ellipse
            # Draw ellipse in random color
            if self.pm :
                cv2.ellipse(image04, ellipse,
                            [random.randrange(256),
                             random.randrange(256),
                            random.randrange(256)], 2)

        # Show image. HighGUI use.
        if self.pm:
            cv2.imshow( "Result", image04 )
        return res # return is a statement, not a function

def fitEllipse(img, slider_pos=40, print_mode=False, color=None):
    if print_mode:
        cv2.namedWindow("Source", 1)
        cv2.namedWindow("Result", 1)
        cv2.resizeWindow('Result', 600,600)
        cv2.imshow("Source", img)
    fe = FitEllipse(img, print_mode, slider_pos, color).process_image(slider_pos)
    if print_mode:
        print("Press any key to exit")
        cv2.waitKey(0)
        cv2.destroyWindow("Source")
        cv2.destroyWindow("Result")
    return fe

# This should be a docstring
"""
Delonay : après avoir trouvé les ellipses, on prend leur centres et on les
relie pour trouver delonay = dual de Voronoi
"""

def Delaunay_perso(img, ellipses, print_mode=False):
    ## What is the structure of the ellipses dictionnary?
    # It should be detailed in the docstring
    list_ellipses_keys = list(ellipses.keys())
    points = np.array([i[0] for i in ellipses.values()])
    #print(points)
    tri = Delaunay(points)
    #print(tri)
    # It is a bit strange to have to access data as the
    # index of the list of keys of a dictionnary
    # Maybe the data structure of your ellipse object is
    # not ideal. If the keys are integers, you can maybe
    # use an array or a DataFrame to store the data
    # code as list_ellipses_keys[i[n]]][0][0] is hard ot read
    # and debug
    # also there is code to plot the triangles in matplotlib
    # that might do that more explicitly
    if print_mode:
        simplices = tri.simplices
        #print(simplices)
        for i in simplices:
            ## Stick to 4 spaces indentations
            for n in range(len(i)):
                A = (round(ellipses[list_ellipses_keys[i[n]]][0][0]),
                     round(ellipses[list_ellipses_keys[i[n]]][0][1]))
                # It is a bad sign that you have to de 3 nested
                # for loops, it can get very slow very fast;
                # there should be a workaround
                for m in range(n+1, len(i)):
                    B = (round(ellipses[list_ellipses_keys[i[m]]][0][0]),
                         round(ellipses[list_ellipses_keys[i[m]]][0][1]))
                    cv2.line(img, A, B, [255,255,255],  1)
        cv2.imshow( "Delonay", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Delonay")
    return(tri)

def Voronoi_perso(img, ellipses, print_mode = 0, d_tri = None):
    if d_tri == None :
        #list_ellipses_keys = list(ellipses.keys())
        points = np.array([ellipse[0] for ellipse in ellipses.values()])
        for i in points :
            cv2.circle(img, (int(round(i[0])), int(round(i[1]))),
                       3, [255,255,255],  1)
        vor = Voronoi(points)
        if print_mode :
            vertices = vor.vertices
            #print(simplices)
            for region in vor.regions:
                if -1 in region:
                    continue
                for i in range(len(region)) :
                    A = (int(round(vertices[region[i]][0])),
                         int(round(vertices[region[i]][1])))
                    B = (int(round(vertices[region[(i+1)%len(region)]][0])),
                         int(round(vertices[region[(i+1)%len(region)]][1])))
                    cv2.line(img, A, B, [255,255,255],  1)
            cv2.imshow( "Voronoi", img)
            cv2.waitKey(0)
            cv2.destroyWindow("Voronoi")
    return(vertices)
