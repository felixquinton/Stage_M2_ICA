#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:34:38 2018

@author: root


Some general remarks:

* There are (quite strict) style guides for python code, that
one should follow. One of those is that the code and comments
should be written in English, as you are not sure only French
speaking persons will read it.

As for the other ones, you can refer to http://pep8.org
You can have your code checked automatically for compliance
with pyflakes and pylint (look for their integration in your IDE)

The rules themselves are followed by programming advices that are good to
follow...

* In python, you should never have to do
 `for i in range(len(sequence))`

If you only need the elements, do `for element in sequence`
If you are sure you need the index of the element, do
`for i, element in enumerate(sequence)` - but that should be rare

* You should avoid to ue global variables (e.g. o_x and o_y here)
It is not very safe, nor easy to modifiy


Also try:

>>> import this

"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import rendering_functions as rdr
from scipy.spatial.distance import cosine


"""
Create a cell image
"""
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def createCell(rads, zeros):
    rd = 1
    cpt = 0 # cpt is a randomly initiated countdown determining where to draw a cell nucleus
    for radian in rads:

        #condition to close the membrane
        if radian >= np.pi/2.0-0.05 :
            if rd > abs(radian - np.pi/2.0) + 1 or rd < -abs(radian - np.pi/2.0) + 1 :
                if rd > 1 : rd = 1+np.pi/2.0-radian
                else : rd = 1-np.pi/2.0+radian
        #solving the equation to determine the coordinate of the point
        x, y = pol2cart(2*rd*30*np.sqrt(2)*np.cos(radian-np.pi/4.0),radian)
        x += 70
        y += 70

        # convert directly x and y to int
        x = int(round(x))
        y = int(round(y))

        #coloriage
        # Look at slicing in numpy
        # https://docs.scipy.org/doc/numpy-dev/user/quickstart.html#indexing-with-arrays-of-indices
        zeros[x-1: x+1,y-1: y+1] = [255, 0, 0]

        #taking the symetric of the current point so that we draw only half of the circle
        s_x = 200-x
        s_y = 200-y
        #coloring
        zeros[s_x-1: s_x+1,s_y-1: s_y+1] = [255, 0, 0]

        #doing the same for the external membrane
        xx,yy = pol2cart(2*rd*60* np.sqrt(2)
                               * np.cos(radian - np.pi/4.0),
                         radian)
        xx += 40
        yy += 40
        xx = int(round(xx))
        yy = int(round(yy))
        zeros[xx-2: xx+2,yy-2: yy+2] = [255, 0, 0]

        #coloring
        s_xx = 200-xx
        s_yy = 200-yy
        #coloring
        zeros[s_xx-2: s_xx+2,s_yy-2: s_yy+2] = [255, 0, 0]

        #updating the random walk
        rd = min(1.08, max(0.92, rd+0.01*(-1+2*(0.5>np.random.rand()))))
        #creating a nucleus
        if cpt == 0:
            center_x = (x + xx) // 2
            center_y = (y + yy) // 2
            #coloring
            zeros[center_x-2: center_x+2,center_y-2: center_y+2] = [0,0,255]
            #symetric
            zeros[200-center_x-2: 200-center_x+2,200-center_y-2: 200-center_y+2] = [0, 0, 255]
            #randomly choosing where the next nucleus will be
            cpt = np.random.randint(10,15)
        cpt -= 1
    return(zeros)

#imgCell = createCell(np.arange(0, (np.pi/2.0), 0.01), np.zeros((200,200,3), dtype=np.uint8))
#saving
#plt.axis('off')
#plt.imshow(imgCell, origin=0)
#plt.savefig('seek4.png')

#import data with opencv
img = cv.imread('seek4.png')


"""
red pixels = membrane
"""
img_red = rdr.isolate_color(img,[0, 0, 255])

# Usually, the different colors will be in well separated
# channels - colors are really just for convenience
# so, instead of the isolate_color code
# you can simply use the grey level channel

"""
not working
img_red = img[:, 0]
"""

red_cont, red_hier = rdr.find_contours(img_red,0)

outside = [contour for contour, hierarchy
          in zip(red_cont, red_hier[0])
          if hierarchy[3] == 0][0]


inside = [contour for contour, hierarchy
          in zip(red_cont, red_hier[0])
          if hierarchy[3] == 2][0]


cv.drawContours(img_red, outside, -1, [255, 0, 0],1)
cv.drawContours(img_red, inside, -1, [255, 0, 0],1)

(o_x,o_y), o_radius = cv.minEnclosingCircle(outside)
(i_x,i_y), i_radius = cv.minEnclosingCircle(inside)

"""
blue pixels = nuclei
"""
img_blu = rdr.isolate_color(img,[255, 0, 0])
blu_cont, blu_hier = rdr.find_contours(img_blu,0)
ellipses = rdr.fitEllipse(img_blu,0,0)
centers = [ellipses[i][0] for i in ellipses]

for ellipse in ellipses:
    cv.ellipse(img, ellipses[ellipse], [np.random.randint(0,256),
                             np.random.randint(0,256),
                             np.random.randint(0,256)], 2)

for center in centers:
    cv.circle(img,(int(center[0]),int(center[1])),1,(0,255,0),1)

"""
Computing the angle between two nuclei to find the bissectrix
"""

#Computing the angle between two points
def angle(v, w):
    # o_x and o_y are the coordinates of the center of the minimum enclosing circle
    v = np.asarray(v) - np.array([o_x, o_y])
    w = np.asarray(w) - np.array([o_x, o_y])
    cos_dist = np.dot(v,w) / (np.sqrt(v[0]**2+v[1]**2)*np.sqrt(w[0]**2+w[1]**2))

    ## BTW This is implemented in scipy as the cosine distance
    #cos_dist = cosine(v, w)
    rad = np.arccos(cos_dist)
    return rad


#Computing the nearest clockwise neighbouring nucleus 
def findNeighbor(cells):
    res = {}
    # nested loops are bad and should be avoided
    # you can look into np.meshgrid to have a flat iterator
    # over a 2D grid

    # Here, you should rather look into
    # the scipy.spatial module, and the
    # KDTree neighbor finding structure
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

    for i in cells:
        ii = (i[0]-o_x,i[1]-o_y)
        min1 = 10**6
        argmin1 = -1
        for j in cells:
            jj = (j[0]-o_x,j[1]-o_y)
            if not ii[0]==jj[0] and not ii[1]==jj[1]:
                if ii[0]*jj[1]-ii[1]*jj[0] > 0 and np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)<min1:
                    min1 = np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)
                    argmin1 = j
        res[i] = argmin1
    return res

neighbors = findNeighbor(centers)

# You can try to refactor the remaining code
# by putting everything in functions, and try to
# facor as much of it as possible and make it more legible

    
def find2FirstVertices(cell):
    #Compute the cutting angle
    cutAngle = angle(cell,neighbors[cell])/2.0
    #Compute the cell coordinate with respect to the center of the minimum enclosing circle
    aa_x, aa_y = (cell[0]-o_x,cell[1]-o_y)
    #Finding a point on the bissectrix
    newPoint = (aa_x * np.cos(cutAngle) - aa_y * np.sin(cutAngle),
                     aa_x * np.sin(cutAngle) + aa_y * np.cos(cutAngle))
    #Compute the new point coordinates wrt the origin
    newPoint = (newPoint[0] + o_x, newPoint[1] + o_y)
    #We have everything needed to compute the intersection points between the circles and the bissectrix :
    #Intersection point on the inside boundary
    distance = np.sqrt((newPoint[0]-i_x)**2+(newPoint[1]-i_y)**2)
    coefCos, coefSin = ((i_radius/(distance))*(cell[0]-i_x),
                        (i_radius/(distance))*(cell[1]-i_y))
    #Coordinate with respect to the center of the minimum enclosing circle
    innerVTPoint = (coefCos*np.cos(cutAngle)-coefSin*np.sin(cutAngle),
                    coefCos*np.sin(cutAngle)+coefSin*np.cos(cutAngle))
    #Retrieve the coordinates wrt the origin
    innerVTPoint = (innerVTPoint[0]+i_x,innerVTPoint[1]+i_y)
    #Intersection point on the outside boundary
    distance = np.sqrt((newPoint[0]-o_x)**2+(newPoint[1]-o_y)**2)
    coefCos, coefSin = ((o_radius/(distance))*(cell[0]-o_x),
                        (o_radius/(distance))*(cell[1]-o_y))
    outerVTPoint = (coefCos*np.cos(cutAngle)-coefSin*np.sin(cutAngle),
                    coefCos*np.sin(cutAngle)+coefSin*np.cos(cutAngle))
    outerVTPoint = (outerVTPoint[0]+o_x,outerVTPoint[1]+o_y)
    
    return [innerVTPoint,outerVTPoint]

VT = {}

for neighbor in neighbors:
    VT[neighbor] = find2FirstVertices(neighbor)
for neighbor in neighbors:
    VT[neighbor].append(VT[neighbors[neighbor]][0])
    VT[neighbor].append(VT[neighbors[neighbor]][1])


"""
Display
"""
def display():
    image0 = img.copy()
    cv.imshow('cell',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #Minimum enclosing circles
    cv.circle(image0,(int(o_x),int(o_y)),int(o_radius),(0,255,0),2)
    cv.circle(image0,(int(i_x),int(i_y)),int(i_radius),(0,255,0),2)
    cv.imshow('cell with minimum enclosing circles',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #Bissectrix
    for neighbor in neighbors:
        cv.circle(image0,(int(round(VT[neighbor][0][0])),
                          int(round(VT[neighbor][0][1]))),2,[255,255,0],1)
        cv.circle(image0,(int(round(VT[neighbor][1][0])),
                          int(round(VT[neighbor][1][1]))),2,[255,255,0],1)
        #cv.circle(img,(int(round(new_point[0])),int(round(new_point[1]))),2,[255,255,0],1)
        cv.line(image0,(int(round(VT[neighbor][1][0])),
                        int(round(VT[neighbor][1][1]))),(
                                int(round(o_x)),
                                int(round(o_y))),[255,255,0],1)
    cv.imshow('cell with bisectrix',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #Voronoi tesselation
    image1 = img.copy()
    for neighbor in neighbors:
        cv.line(image1,(int(round(VT[neighbor][0][0])),
                        int(round(VT[neighbor][0][1]))),(
                                int(round(VT[neighbor][1][0])),
                                int(round(VT[neighbor][1][1]))),[255,255,255],1)
        cv.line(image1,(int(round(VT[neighbor][0][0])),
                        int(round(VT[neighbor][0][1]))),(
                                int(round(VT[neighbor][2][0])),
                                int(round(VT[neighbor][2][1]))),[255,255,255],1)
        cv.line(image1,(int(round(VT[neighbor][1][0])),
                        int(round(VT[neighbor][1][1]))),(
                                int(round(VT[neighbor][3][0])),
                                int(round(VT[neighbor][3][1]))),[255,255,255],1)

    cv.imshow('cell with VT',image1)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()

display()

"""
Check if the cells are polarised by computing the distance from their nucleus to their boundaries
"""
#d1 and d2 are the points defining the boundary. a is the point to project  on (d1,d2)
def orthoProj2D(d1,d2,a):
    X = ((d2[0]-d1[0])*(a[0]-d2[0])+(d2[1]-d1[1])*(a[1]-d2[1]))/((d2[0]-d1[0])**2+(d2[1]-d1[1])**2)
    res = (d2[0]+(d2[0]-d1[0])*X,d2[1]+(d2[1]-d1[1])*X)
    return res
image1 = img.copy()

isPolar = {}
dicOrthProj = {}

for center in centers:
    p_i = orthoProj2D(VT[center][0],VT[center][2],center)
    #cv.circle(image1,(int(round(p_i[0])),int(round(p_i[1]))),2,[255,255,0],1)
    p_o = orthoProj2D(VT[center][1],VT[center][3],center)
    cv.circle(image1,(int(round(p_o[0])),int(round(p_o[1]))),2,[255,255,0],1)
    d_i = np.sqrt((p_i[0]-center[0])**2+(p_i[1]-center[1])**2)
    d_o = np.sqrt((p_o[0]-center[0])**2+(p_o[1]-center[1])**2)
    isPolar[center] = (d_i>d_o)
    dicOrthProj[center] = (p_i,p_o)

print("Proportion of nuclei closest to the outside = ", sum([isPolar[i] for i in isPolar])/len(isPolar))

"""
Compute the orientation of the nuclei wrt the outer boundary
"""

#for c in ell :
#    if ell[c][1][0]<ell[c][1][0]:o = ell[c][2]+90
#    else : o = ell[c][2]
#    nc = ell[c][0]
#    m = (VT[nc][1],VT[nc][3])
#    print(m)
#    cosx = (VT[nc][3])/(np.sqrt(VT[nc][3]**2+VT[nc][1]**2))
