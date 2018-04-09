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
from sympy.solvers import solve
from sympy import Symbol, cos
import rendering_functions as rdr
from scipy.spatial.distance import cosine


"""
Créer une image de cellule
"""
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


# those should be arguments of the function
rads = np.arange(0, (np.pi/2.0), 0.01)
rd = 1 # What's rd?
zeros = np.zeros((200,200,3),
                 dtype=np.uint8)
r = Symbol('r')
theta = Symbol('theta')
cpt = 0 # what's cpt
#boucle sur l'angle

def create_cell():
    rd = 1
    for radian in rads:

        #condition pour fermer la membrane
        if radian >= np.pi/2.0-0.05 :
            if rd > abs(radian - np.pi/2.0) + 1 or rd < -abs(radian - np.pi/2.0) + 1 :
                if rd > 1 : rd = 1+np.pi/2.0-radian
                else : rd = 1-np.pi/2.0+radian
        #résolution de l'équation pour déterminer les coordonnées polaires du point du cercle
        x, y = pol2cart(solve(r-2*rd*30*np.sqrt(2)*cos(radian-np.pi/4.0))[0],radian)
        x += 70
        y += 70

        # convert directly x and y to int
        x = int(round(x))
        y = int(round(y))

        #coloriage
        # Look at slicing in numpy
        # https://docs.scipy.org/doc/numpy-dev/user/quickstart.html#indexing-with-arrays-of-indices
        zeros[x-1: x+1][y-1: y+1] = [255, 0, 0]

        #on prend son symétrique pour aller plus vite en ne faisant que la moitié du cercle
        s_x = 200-x
        s_y = 200-y
        #coloriage
        zeros[s_x-1: s_x+1][s_y-1: s_y+1] = [255, 0, 0]

        #on fait pareil pour la membrane extérieure
        xx,yy = pol2cart(solve(r-2*rd*60* np.sqrt(2)
                               * cos(radian - np.pi/4.0))[0],
                         radian)
        xx += 40
        yy += 40
        xx = int(round(xx))
        yy = int(round(yy))
        zeros[xx-1: xx+1][yy-1: yy+1] = [255, 0, 0]

        #coloriage
        s_xx = 200-xx
        s_yy = 200-yy
        #coloriage
        zeros[s_xx-1: s_xx+1][s_yy-1: s_yy+1] = [255, 0, 0]

        #on relance la marche aléatoire à chaque itération
        rd = min(1.08, max(0.92, rd+0.01*(-1+2*(0.5>np.random.rand()))))
        #création d'un noyau
        if cpt == 0:
            center_x = (x + xx) // 2
            center_x = (y + yy) // 2
            #coloriage
            zeros[center_x-2: center_x+2][center_y-2: center_y+2]
            #symétrique
            zeros[200-center_x-2: 200-center_y-2][200-center_y-2: 200-center_y+2] = [0, 0, 255]
            #relance du cpt pour déterminer où sera le prochain noyau
            cpt = np.random.randint(10,15)
        cpt -= 1 # ?

"""
create_cell()
#enregistrement
plt.axis('off')
plt.imshow(zeros, origin=0)
plt.savefig('seek4.png')
"""
#récupération avec opencv
img = cv.imread('seek4.png')


"""
traitement des pixels rouges = membrane
"""
img_red = rdr.isolate_color(img,[0, 0, 255])

# Usually, the different colors will be in well separated
# channels - colors are really just for convenience
# so, instead of the isolate_color code
# you can simply use the grey level channel
img_red = img[:, 0]

red_cont, red_hier = rdr.find_contours(img_red,0)

ouside = [contour for contour, hierarchy
          in zip(red_cont[0], red_hier[0])
          if hierarchy[3] == 0]

# I leace the remaining range(len()) to be removed as an exercise

inside = [red_cont[i] for i in range(len(red_hier[0])) if red_hier[0][i][3]==2]

cv.drawContours(img_red, outside, -1, [255, 0, 0],1)
cv.drawContours(img_red, inside, -1, [255, 0, 0],1)

outside = [list(outside[0][i][0]) for i in range(len(outside[0]))]
inside = [list(inside[0][i][0]) for i in range(len(inside[0]))]
(o_x,o_y), o_radius = cv.minEnclosingCircle(np.array(outside))
(i_x,i_y), i_radius = cv.minEnclosingCircle(np.array(inside))

"""
traitement des pixels bleus = noyau
"""
img_blu = rdr.isolate_color(img,[255, 0, 0])
blu_cont, blu_hier = rdr.find_contours(img_blu,0)
ell = rdr.fitEllipse(img_blu,0,0)
nuc_center = [ell[i][0] for i in ell]

for e in ell:
    cv.ellipse(img, ell[e], [np.random.randint(0,256),
                             np.random.randint(0,256),
                             np.random.randint(0,256)], 2)

for i in nuc_center:
    cv.circle(img,(int(i[0]),int(i[1])),1,(0,255,0),1)

"""
Calcul de l'angle entre deux noyau pour trouver la bissectrice
"""

#calcul l'angle à l'origine entre deux points
def angle(v, w):
    # This was far too hard to read
    # you can simplify by refactoring:
    # Where are o_x and o_y defined?
    v = np.asarray(v) - np.array([o_x, o_y])
    w = np.asarray(w) - np.array([o_x, o_y])
    cos_dist = (v * w).sum() / (np.linalg.norm(v) * np.linalg.norm(w))

    ## BTW This is implemented in scipy as the cosine distance
    cos_dist = cosine(v, w)
    rad = np.arccos(cos_dist)
    return rad


#calcul du noyau le plus proche pour chaque noyau (dans le sens des aiguilles d'une montre)
def voisins(list):
    # list is a reserved word, you should _not_ use it as
    # a variable name!!
    res = {}
    # nested loops are bad and should be avoided
    # you can look into np.meshgrid to have a flat iterator
    # over a 2D grid

    # Here, you should rather look into
    # the scipy.spatial module, and the
    # KDTree neighbor finding structure
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html

    for i in list:
        ii = (i[0]-o_x,i[1]-o_y)
        min1 = 10**6
        argmin1 = -1
        for j in list:
            jj = (j[0]-o_x,j[1]-o_y)
            if not ii[0]==jj[0] and not ii[1]==jj[1]:
                if ii[0]*jj[1]-ii[1]*jj[0] > 0 and np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)<min1:
                    min1 = np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)
                    argmin1 = j
        res[i] = argmin1
    return(res)

voisin = voisins(nuc_center)
tmp_vt = {}

# I usually defined lists variables with an 's' ath the
# end of the name, so that we now it's a collection
# so `voisin` should be named `neighbors`
# and you can do
# for neighbor in neighbors:
#     ...

# You can try to refactor the remaining code
# by putting everything in functions, and try to
# facor as much of it as possible and make it more legible


for a in voisin:
    #on récupère l'angle pour former la bissectrice
    angle_to_cut = angle(a,voisin[a])
    m_angle = angle_to_cut/2.0
    #on récupère les coordonnées de a par rapport au centre de l'organoïde
    aa_x, aa_y = (a[0]-o_x,a[1]-o_y)
    #on trouve les coordonnées d'un point pour définir la bissectrice
    new_point_tmp = (aa_x * np.cos(m_angle) - aa_y * np.sin(m_angle),
                     aa_x * np.sin(m_angle) + aa_y * np.cos(m_angle))
    #on récupère les coordonnées du nouveau point par rapport à l'origine
    new_point = (new_point_tmp[0] + o_x, new_point_tmp[1] + o_y)
    #on calcule les points d'intersection entre la bissectrice et les frontières de l'organoïde
    ## Exercice : make this legible ;)
    #inside
    d = np.sqrt((new_point[0]-i_x)**2+(new_point[1]-i_y)**2)
    i_b_x, i_b_y = ((i_radius/(d))*(a[0]-i_x),(i_radius/(d))*(a[1]-i_y))
    n_p_i = (i_b_x*np.cos(m_angle)-i_b_y*np.sin(m_angle),i_b_x*np.sin(m_angle)+i_b_y*np.cos(m_angle))
    n_p_i = (n_p_i[0]+i_x,n_p_i[1]+i_y)
    #outside
    d = np.sqrt((new_point[0]-o_x)**2+(new_point[1]-o_y)**2)
    o_b_x, o_b_y = ((o_radius/(d))*(a[0]-o_x),(o_radius/(d))*(a[1]-o_y))
    n_p_o = (o_b_x*np.cos(m_angle)-o_b_y*np.sin(m_angle),o_b_x*np.sin(m_angle)+o_b_y*np.cos(m_angle))
    n_p_o = (n_p_o[0]+o_x,n_p_o[1]+o_y)

    tmp_vt[a] = [n_p_i,n_p_o]



ttmp_vt = {}
for a in voisin:
    ttmp_vt[a] = tmp_vt[voisin[a]]
#on sauvegarde quatres points par noyau pour la VT
VT = {}
for a in voisin :
    VT[a] = [i for i in tmp_vt[a]+ttmp_vt[a]]


"""
Display
"""
def display():
    image0 = img.copy()
    cv.imshow('cell',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #cercles
    cv.circle(image0,(int(o_x),int(o_y)),int(o_radius),(0,255,0),2)
    cv.circle(image0,(int(i_x),int(i_y)),int(i_radius),(0,255,0),2)
    cv.imshow('cell with minimum enclosing circles',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #bissectrices
    for a in voisin:
        cv.circle(image0,(int(round(VT[a][0][0])),int(round(VT[a][0][1]))),2,[255,255,0],1)
        cv.circle(image0,(int(round(VT[a][1][0])),int(round(VT[a][1][1]))),2,[255,255,0],1)
        #cv.circle(img,(int(round(new_point[0])),int(round(new_point[1]))),2,[255,255,0],1)
        cv.line(image0,(int(round(VT[a][1][0])),int(round(VT[a][1][1]))),(int(round(o_x)),int(round(o_y))),[255,255,0],1)
    cv.imshow('cell with bisectrix',image0)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()
    #VT
    image1 = img.copy()
    for a in VT:
        cv.line(image1,(int(round(VT[a][0][0])),int(round(VT[a][0][1]))),(int(round(VT[a][1][0])),int(round(VT[a][1][1]))),[255,255,255],1)
        cv.line(image1,(int(round(VT[a][0][0])),int(round(VT[a][0][1]))),(int(round(VT[a][2][0])),int(round(VT[a][2][1]))),[255,255,255],1)
        cv.line(image1,(int(round(VT[a][1][0])),int(round(VT[a][1][1]))),(int(round(VT[a][3][0])),int(round(VT[a][3][1]))),[255,255,255],1)

    cv.imshow('cell with VT',image1)
    k = cv.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv.destroyAllWindows()

display()

"""
Vérifier si les cellules sont polarisées en mesurant leur distance à la frontière basale et à la
la frontière apicale
"""
#d1 et d2 sont les points qui définissent la droite en 2D. a est le point dont on calcule le projeté
def orthoProj2D(d1,d2,a):
    X = ((d2[0]-d1[0])*(a[0]-d2[0])+(d2[1]-d1[1])*(a[1]-d2[1]))/((d2[0]-d1[0])**2+(d2[1]-d1[1])**2)
    res = (d2[0]+(d2[0]-d1[0])*X,d2[1]+(d2[1]-d1[1])*X)
    return(res)
image1 = img.copy()

isPolar = {}
dicOrthProj = {}

for c in nuc_center:
    p_i = orthoProj2D(VT[c][0],VT[c][2],c)
    #cv.circle(image1,(int(round(p_i[0])),int(round(p_i[1]))),2,[255,255,0],1)
    p_o = orthoProj2D(VT[c][1],VT[c][3],c)
    cv.circle(image1,(int(round(p_o[0])),int(round(p_o[1]))),2,[255,255,0],1)
    d_i = np.sqrt((p_i[0]-c[0])**2+(p_i[1]-c[1])**2)
    d_o = np.sqrt((p_o[0]-c[0])**2+(p_o[1]-c[1])**2)
    isPolar[c] = (d_i>d_o)
    dicOrthProj[c] = (p_i,p_o)

print("Proportion de noyau plus proche de l'extérieur = ", sum([isPolar[i] for i in isPolar])/len(isPolar))

"""
Calculer l'orientation du noyau des cellules par rapport à la frontière extérieure
"""

#for c in ell :
#    if ell[c][1][0]<ell[c][1][0]:o = ell[c][2]+90
#    else : o = ell[c][2]
#    nc = ell[c][0]
#    m = (VT[nc][1],VT[nc][3])
#    print(m)
#    cosx = (VT[nc][3])/(np.sqrt(VT[nc][3]**2+VT[nc][1]**2))
