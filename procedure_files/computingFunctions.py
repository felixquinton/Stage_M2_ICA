#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:50:24 2018

@author: fquinton
"""

import numpy as np

"""
Computing the angle between two nuclei to find the bissectrix
"""

#Computing the angle between two points
def angle(v, w, o):
    # o_x and o_y are the coordinates of the center of the minimum enclosing circle
    v = np.asarray(v) - np.array(o)
    w = np.asarray(w) - np.array(o)
    cos_dist = np.dot(v,w) / (np.sqrt(v[0]**2+v[1]**2)*np.sqrt(w[0]**2+w[1]**2))

    ## BTW This is implemented in scipy as the cosine distance
    #cos_dist = cosine(v, w)
    rad = np.arccos(cos_dist)
    return rad


#Computing the nearest clockwise neighbouring nucleus 
def findNeighbor(cells, o):
    res = {}
    # nested loops are bad and should be avoided
    # you can look into np.meshgrid to have a flat iterator
    # over a 2D grid

    # Here, you should rather look into
    # the scipy.spatial module, and the
    # KDTree neighbor finding structure
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.cKDTree.html
    for i in cells:
        ii = (i[0]-o[0],i[1]-o[1])
        min1 = 10**6
        argmin1 = -1
        for j in cells:
            jj = (j[0]-o[0],j[1]-o[1])
            if not ii[0]==jj[0] and not ii[1]==jj[1]:
                if ii[0]*jj[1]-ii[1]*jj[0] > 0 and np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)<min1:
                    min1 = np.sqrt((ii[0]-jj[0])**2+(ii[1]-jj[1])**2)
                    argmin1 = j
        res[i] = argmin1
    return res

def find_closer_angle(theta0, theta1):
    '''Finds the index of the closest value in theta1
    for each value in theta0, with 2π periodic boundary
    conditions.
    
    Parameters
    ----------
    theta0 : np.ndarray of shape (N0,)
      the target values 
    theta1 : np.ndarray of shape (N1,)
      array where we search for the values closest
      to the targer theta0
      
    Returns
    -------
    indices : nd.array of shape (N0,)
      the indices of the values closest to theta0 in theta1
      
    Example
    -------
    >>> theta0 = np.array([0, 0.5, 0.79])*2*np.pi
    >>> theta1 = np.array([0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.])*2*np.pi
    >>> find_closer_angle(theta0, theta1)
        np.array([0, 4, 5])
    '''
    tt0, tt1 = np.meshgrid(theta0, theta1)   
    dtheta = tt0 - tt1
    # periodic boundary
    dtheta[dtheta >   np.pi] -= 2*np.pi
    dtheta[dtheta <= -np.pi] += 2*np.pi
    
    return (dtheta**2).argmin(axis=0)

def get_bissecting_vertices(centers, inners, outers):
    '''Docstring left as an exercice
    '''
    theta_centers = np.arctan2(centers[:, 1], centers[:, 0])
    bissect = (theta_centers + np.roll(theta_centers, 1, axis=0))/2
    dtheta = (theta_centers - np.roll(theta_centers, 1, axis=0))
    # periodic boundary
    bissect[dtheta >= np.pi] -= np.pi
    bissect[dtheta < -np.pi] += np.pi
    theta_inners = np.arctan2(inners[:, 1], inners[:, 0])
    theta_outers = np.arctan2(outers[:, 1], outers[:, 0])

    inner_vs = inners.take(find_closer_angle(bissect, theta_inners), axis=0)
    outer_vs = outers.take(find_closer_angle(bissect, theta_outers), axis=0)
    return inner_vs, outer_vs