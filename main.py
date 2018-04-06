#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:34:38 2018

@author: root
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from sympy.solvers import solve
from sympy import Symbol, cos
import sys 
import os
sys.path.append(os.path.abspath('/home/fquinton/Documents/stage_cell'))
import rendering_functions as rdr


"""
Créer une image de cellule
"""
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)
    
rads = np.arange(0, (np.pi/2.0), 0.01)
rd = 1
zeros = np.zeros((200,200,3),dtype = np.uint8)
r = Symbol('r')
theta = Symbol('theta')
cpt = 0
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
        x,y = pol2cart(solve(r-2*rd*30*np.sqrt(2)*cos(radian-np.pi/4.0))[0],radian)
        x += 70
        y += 70
        #coloriage
        zeros[round(x)][round(y)]=[255,0,0]
        zeros[round(x)-1][round(y)]=[255,0,0]
        zeros[round(x)+1][round(y)]=[255,0,0]
        zeros[round(x)][round(y)-1]=[255,0,0]
        zeros[round(x)][round(y)+1]=[255,0,0]
        zeros[round(x)-1][round(y)-1]=[255,0,0]
        zeros[round(x)+1][round(y)+1]=[255,0,0]
        zeros[round(x)+1][round(y)-1]=[255,0,0]
        zeros[round(x)-1][round(y)+1]=[255,0,0]
        #on prend son symétrique pour aller plus vite en ne faisant que la moitié du cercle
        s_x = 200-x
        s_y = 200-y
        #coloriage
        zeros[round(s_x)][round(s_y)]=[255,0,0]
        zeros[round(s_x)-1][round(s_y)]=[255,0,0]
        zeros[round(s_x)+1][round(s_y)]=[255,0,0]
        zeros[round(s_x)][round(s_y)-1]=[255,0,0]
        zeros[round(s_x)][round(s_y)+1]=[255,0,0]
        zeros[round(s_x)-1][round(s_y)-1]=[255,0,0]
        zeros[round(s_x)+1][round(s_y)+1]=[255,0,0]
        zeros[round(s_x)+1][round(s_y)-1]=[255,0,0]
        zeros[round(s_x)-1][round(s_y)+1]=[255,0,0]
        #on fait pareil pour la membrane extérieure
        xx,yy = pol2cart(solve(r-2*rd*60*np.sqrt(2)*cos(radian- np.pi/4.0))[0],radian)
        xx += 40
        yy += 40
        #coloriage
        zeros[round(xx)][round(yy)]=[255,0,0]
        zeros[round(xx)-1][round(yy)]=[255,0,0]
        zeros[round(xx)+1][round(yy)]=[255,0,0]
        zeros[round(xx)][round(yy)-1]=[255,0,0]
        zeros[round(xx)][round(yy)+1]=[255,0,0]
        zeros[round(xx)-1][round(yy)-1]=[255,0,0]
        zeros[round(xx)+1][round(yy)+1]=[255,0,0]
        zeros[round(xx)+1][round(yy)-1]=[255,0,0]
        zeros[round(xx)-1][round(yy)+1]=[255,0,0]
        s_xx = 200-xx
        s_yy = 200-yy
        #coloriage
        zeros[round(s_xx)][round(s_yy)]=[255,0,0]
        zeros[round(s_xx)-1][round(s_yy)]=[255,0,0]
        zeros[round(s_xx)+1][round(s_yy)]=[255,0,0]
        zeros[round(s_xx)][round(s_yy)-1]=[255,0,0]
        zeros[round(s_xx)][round(s_yy)+1]=[255,0,0]
        zeros[round(s_xx)-1][round(s_yy)-1]=[255,0,0]
        zeros[round(s_xx)+1][round(s_yy)+1]=[255,0,0]
        zeros[round(s_xx)+1][round(s_yy)-1]=[255,0,0]
        zeros[round(s_xx)-1][round(s_yy)+1]=[255,0,0]
        #on relance la marche aléatoire à chaque itération
        rd = min(1.08,max(0.92,rd + 0.01*(-1+2*(0.5>np.random.rand()))))
        #création d'un noyau
        if cpt == 0:
            #coloriage
            zeros[round((x+xx)/2)][round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)+1][round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)-1][round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)][1+round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)][-1+round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)+1][round((y+yy)/2)+1] = [0,0,255]
            zeros[round((x+xx)/2)-1][round((y+yy)/2)+1] = [0,0,255]
            zeros[round((x+xx)/2)+1][round((y+yy)/2)-1] = [0,0,255]
            zeros[round((x+xx)/2)-1][round((y+yy)/2)-1] = [0,0,255]
            zeros[round((x+xx)/2)+2][round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)-2][round((y+yy)/2)] = [0,0,255]
            zeros[round((x+xx)/2)][round((y+yy)/2)-2] = [0,0,255]
            zeros[round((x+xx)/2)][round((y+yy)/2)+2] = [0,0,255]
            zeros[round((x+xx)/2)+2][round((y+yy)/2)+1] = [0,0,255]
            zeros[round((x+xx)/2)-2][round((y+yy)/2)-1] = [0,0,255]
            zeros[round((x+xx)/2)+1][round((y+yy)/2)-2] = [0,0,255]
            zeros[round((x+xx)/2)-1][round((y+yy)/2)+2] = [0,0,255]
            #symétrique
            zeros[200-round((x+xx)/2)][200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)+1][200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)-1][200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)][1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)][-1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)+1][200-round((y+yy)/2)+1] = [0,0,255]
            zeros[200-round((x+xx)/2)-1][200-round((y+yy)/2)+1] = [0,0,255]
            zeros[200-round((x+xx)/2)+1][-1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)-1][-1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)+2][-1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)-2][-1+200-round((y+yy)/2)] = [0,0,255]
            zeros[200-round((x+xx)/2)][-1+200-round((y+yy)/2)+2] = [0,0,255]
            zeros[200-round((x+xx)/2)][-1+200-round((y+yy)/2)-2] = [0,0,255]
            zeros[200-round((x+xx)/2)+2][200-round((y+yy)/2)+1] = [0,0,255]
            zeros[200-round((x+xx)/2)-2][200-round((y+yy)/2)-1] = [0,0,255]
            zeros[200-round((x+xx)/2)+1][200-round((y+yy)/2)+2] = [0,0,255]
            zeros[200-round((x+xx)/2)-1][200-round((y+yy)/2)-2] = [0,0,255]
            #relance du cpt pour déterminer où sera le prochain noyau
            cpt = np.random.randint(10,15)
        cpt -= 1
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
img_red = rdr.isolate_color(img,[0,0,255])       

red_cont, red_hier = rdr.find_contours(img_red,0)
outside = [red_cont[i] for i in range(len(red_hier[0])) if red_hier[0][i][3]==0]
inside = [red_cont[i] for i in range(len(red_hier[0])) if red_hier[0][i][3]==2]
cv.drawContours(img_red, outside,-1, [255,0,0],1)
cv.drawContours(img_red, inside,-1, [255,0,0],1)
outside = [list(outside[0][i][0]) for i in range(len(outside[0]))]
inside = [list(inside[0][i][0]) for i in range(len(inside[0]))]
(o_x,o_y),o_radius = cv.minEnclosingCircle(np.array(outside))
(i_x,i_y),i_radius = cv.minEnclosingCircle(np.array(inside))

"""
traitement des pixels bleus = noyau
"""
img_blu = rdr.isolate_color(img,[255,0,0])
blu_cont, blu_hier = rdr.find_contours(img_blu,0)
ell = rdr.fitEllipse(img_blu,0,0)
nuc_center = [ell[i][0] for i in ell]
for e in ell:
    cv.ellipse(img, ell[e], [np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256)], 2)
for i in nuc_center:
    cv.circle(img,(int(i[0]),int(i[1])),1,(0,255,0),1)

"""
Calcul de l'angle entre deux noyau pour trouver la bissectrice
"""
#calcul l'angle à l'origine entre deux points
def angle(v,w):
    rad = np.arccos(((v[0]-o_x)*(w[0]-o_x)+(v[1]-o_y)*(w[1]-o_y))/(np.sqrt((v[0]-o_x)**2+(v[1]-o_y)**2)*np.sqrt((w[0]-o_x)**2+(w[1]-o_y)**2)))
    return(rad)
#calcul du noyau le plus proche pour chaque noyau (dans le sens des aiguilles d'une montre)
def voisins(list):
    res = {}
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
    
for a in voisin:
    #on récupère l'angle pour former la bissectrice
    angle_to_cut = angle(a,voisin[a])
    m_angle = angle_to_cut/2.0
    #on récupère les coordonnées de a par rapport au centre de l'organoïde
    aa_x, aa_y = (a[0]-o_x,a[1]-o_y)
    #on trouve les coordonnées d'un point pour définir la bissectrice
    new_point_tmp = (aa_x*np.cos(m_angle)-aa_y*np.sin(m_angle),aa_x*np.sin(m_angle)+aa_y*np.cos(m_angle))
    #on récupère les coordonnées du nouveau point par rapport à l'origine
    new_point = (new_point_tmp[0]+o_x,new_point_tmp[1]+o_y)
    #on calcule les points d'intersection entre la bissectrice et les frontières de l'organoïde
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