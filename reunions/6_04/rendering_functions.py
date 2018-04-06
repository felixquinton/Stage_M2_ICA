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

"""
Isoler les pixels d'une couleur donnée

img = image cv2
color = couleur à isoler. triplet [x,y,z] code couleur BGR. La recherche est effectuée dans
un interval autour de cette couleur contrôlé par la parametre interval
INV = booléen. si 0 la fonction renvoie une image avec seulement les pixels de la couleur color
si 1 la fonction renvoie une image sans les pixels de la couleur color.
interval = int. largeur de l'interval.optional. défaut = 10.
"""

def isolate_color(img, color, INV=0, interval = 20):
    
    if not INV :
        #Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
        lower= np.array([hsv_color[0][0][0]-interval,100,100])
        upper= np.array([hsv_color[0][0][0]+interval,255,255])
        
        # Threshold the HSV image to get only desired colors
        mask = cv2.inRange(hsv, lower, upper)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        
        return(res)
    #Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_color = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)
    lower= np.array([hsv_color[0][0][0]-interval,100,100])
    upper= np.array([hsv_color[0][0][0]+interval,255,255])
    
    # Threshold the HSV image to get only desired colors
    mask = cv2.inRange(hsv, lower, upper)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(img,img, mask= cv2.bitwise_not(mask))
    
    return(res)
    
    
"""
Trouve les contours d'une image en BGR.
Retourne la liste des contours et le tableau de hierarchie
"""
def find_contours(img,slider_pos):
    imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,slider_pos,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return(contours, hierarchy)
    
"""
Retourne les contours de hierarchie 0 pour une image BGR
"""
    
def hierarchy_clean(img, isGivenContours = 0, contours = None, hierarchy = None):
    if not isGivenContours : contours, hierarchy = find_contours(img)
    res = [contours[i] for i in range(len(hierarchy[0])) if hierarchy[0][i][3]==-1]
    return(res)
    
"""
Retourne le squelette d'une liste de contours. prend en entrée l'image et la lsite de contours
retourne
"""

def skeleton(img, contours):
    pass


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
        if not color == None : self.source_image = isolate_color(self.source_image, color, 0)
        if print_mode : cv2.createTrackbar("Threshold", "Result", slider_pos, 255, self.process_image)
        self.process_image(slider_pos)

    def process_image(self, slider_pos):
        """
        This function finds contours, draws them and their approximation by ellipses.
        """
        # Create the destination images
        if self.pm : image04 = self.source_image.copy()
        contours, hierarchy = find_contours(self.source_image, slider_pos)
        cont = hierarchy_clean(self.source_image, 1, contours, hierarchy)
        res = {}
        for c in cont:
            # Number of points must be more than or equal to 6 for cv.FitEllipse2
            if len(c) >= 6:
                # Draw the current contour in gray
                if self.pm : cv2.drawContours(image04, c,-1, [0,0,250],1)
                # Fits ellipse to current contour.
                ellipse = cv2.fitEllipse(c)
                tuple_c = tuple([tuple(i[0]) for i in c])
                res[tuple_c] = ellipse                
                # Draw ellipse in random color
                if self.pm : cv2.ellipse(image04, ellipse, [random.randrange(256),random.randrange(256),random.randrange(256)], 2)
        # Show image. HighGUI use.
        if self.pm : cv2.imshow( "Result", image04 )
        return(res)

def fitEllipse(img, slider_pos = 40, print_mode = 0, color = None):
    if print_mode == 1:
        cv2.namedWindow("Source", 1)
        cv2.namedWindow("Result", 1)
        cv2.resizeWindow('Result', 600,600)
        cv2.imshow("Source", img)
    fe = FitEllipse(img, print_mode, slider_pos, color).process_image(slider_pos)
    if print_mode == 1:    
        print("Press any key to exit")
        cv2.waitKey(0)    
        cv2.destroyWindow("Source")
        cv2.destroyWindow("Result")
    return(fe)
    
"""
Delonay : après avoir trouver les ellipses, on prend leur centres et on les
relie pour trouver delonay = dual de Voronoi
"""   

def Delaunay_perso(img, ellipses, print_mode = 0):
    list_ellipses_keys = list(ellipses.keys())
    points = np.array([i[0] for i in list(ellipses.values())])
    #print(points)
    tri = Delaunay(points)
    #print(tri)
    if print_mode :
        simplices = tri.simplices
        #print(simplices)
        for i in simplices:
                for n in range(len(i)):
                    A = (round(ellipses[list_ellipses_keys[i[n]]][0][0]),round(ellipses[list_ellipses_keys[i[n]]][0][1]))
                    for m in range(n+1,len(i)): 
                        B = (round(ellipses[list_ellipses_keys[i[m]]][0][0]),round(ellipses[list_ellipses_keys[i[m]]][0][1]))
                        cv2.line(img, A, B, [255,255,255],  1)
        cv2.imshow( "Delonay", img)
        cv2.waitKey(0)    
        cv2.destroyWindow("Delonay")
    return(tri)
    
def Voronoi_perso(img, ellipses, print_mode = 0, d_tri = None):
    if d_tri == None :
        #list_ellipses_keys = list(ellipses.keys())
        points = np.array([i[0] for i in list(ellipses.values())])
        for i in points :
            cv2.circle(img, (int(round(i[0])),int(round(i[1]))), 3, [255,255,255],  1)
        vor = Voronoi(points)
        if print_mode :
            vertices = vor.vertices
            #print(simplices)
            for region in vor.regions:
                if not -1 in region :
                    for i in range(len(region)) :
                        A = (int(round(vertices[region[i]][0])),int(round(vertices[region[i]][1])))
                        B = (int(round(vertices[region[(i+1)%len(region)]][0])),int(round(vertices[region[(i+1)%len(region)]][1])))
                        cv2.line(img, A, B, [255,255,255],  1)
            cv2.imshow( "Voronoi", img)
            cv2.waitKey(0)    
            cv2.destroyWindow("Voronoi")
    return(vertices)