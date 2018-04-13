#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:54:49 2018

@author: fquinton
"""

import numpy as np
import matplotlib.pyplot as plt

#from tyssue.generation import generate_ring
from tyssue import PlanarGeometry
from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.dynamics import effectors, factory
from math import sin
import cv2 as cv
import imageProcessing as iP
import computingFunctions as cpF
import tyssueMissing





#import data with opencv
img = cv.imread('seek3.png')


"""
red pixels = membrane
"""

membraneData = iP.membraneExtraction(img)

#saving useful data on the membrane
outside = membraneData[0]
inside = membraneData[1]
o_x, o_y, o_radius = membraneData[2:5]
i_x,i_y, i_radius = membraneData[5::]

"""
blue pixels = nuclei
"""

nucleiData = iP.nucleiDetection(img)

#saving useful data on the nuclei
ellipses = nucleiData[0]
centers = nucleiData[1]


#for each cell defined by its center, compute its closest clockwise neighbor   
neighbors = cpF.findNeighbor(centers, (o_x,o_y))
    
        
#inside = (inside - np.ones(inside.shape)*(img.shape[0]/2.0))/float(img.shape[0])
#outside = (outside - np.ones(outside.shape)*(img.shape[0]/2.0))/float(img.shape[0])


inside = np.array([[(inside[i][0][0]-img.shape[0]/2.0)/img.shape[0],(inside[i][0][1]-img.shape[1]/2.0)/img.shape[1]] for i in range(len(inside))])
outside = np.array([[(outside[i][0][0]-img.shape[0]/2.0)/img.shape[0],(outside[i][0][1]-img.shape[1]/2.0)/img.shape[1]] for i in range(len(outside))])
#we need the list of cell nuclei to be clockwise ordered
clockwiseCenters = [neighbors[centers[0]]]
for i in centers[0:len(centers)-1] :
    tmp = clockwiseCenters[len(clockwiseCenters)-1]
    clockwiseCenters.append(neighbors[tmp])

clockwiseCenters = np.array([[(clockwiseCenters[i][0]-img.shape[0]/2.0)/img.shape[0],(clockwiseCenters[i][1]-img.shape[1]/2.0)/img.shape[1]] for i in range(len(clockwiseCenters))])


#defining de organoid using the data we saved above
Nf = len(clockwiseCenters)
R_in = i_radius/img.shape[1]
R_out = o_radius/img.shape[1]
inners = inside
outers = outside
centers = clockwiseCenters

#plotting the organoid
fig, ax = plt.subplots()
ax.plot(*inners.T)
ax.plot(*outers.T)
ax.scatter(centers[:, 0], centers[:, 1], marker='+')
_ = ax.set(aspect='equal')


#compute the vertices of the mesh
inner_vs, outer_vs = cpF.get_bissecting_vertices(centers, inners, outers)

#plotting the computed mesh vertices on the organoid image
fig, ax = plt.subplots()
ax.plot(*inners.T)
ax.plot(*outers.T)
ax.scatter(centers[:, 0], centers[:, 1], marker='+')
ax.plot(inner_vs[:, 0], inner_vs[:, 1], '-o', alpha=0.8)
ax.plot(outer_vs[:, 0], outer_vs[:, 1], '-o', alpha=0.8)
_ = ax.set(aspect='equal')


#initialising the mesh
organo = tyssueMissing.generate_ring(Nf, R_in, R_out)

# adjustement of the mesh to the vertices computed above.
organo.vert_df.loc[organo.apical_verts, organo.coords] = inner_vs
organo.vert_df.loc[organo.basal_verts, organo.coords] = outer_vs

# Geometry update
PlanarGeometry.update_all(organo)

# Plot of the mesh
fig, ax = plt.subplots()
ax.plot(*inners.T, lw=3, alpha=0.6)
ax.plot(*outers.T, lw=3, alpha=0.6)

fig, ax = quick_edge_draw(organo, ax=ax)

ax.set(aspect='equal');


# Construction of the model
model = factory.model_factory(
    [effectors.FaceAreaElasticity,
     effectors.LineTension], 
    effectors.FaceAreaElasticity)
print('We build a model with the following terms:')
print('\t', *model.labels, sep='\n\t')


# Model parameters or specifications
specs = {
    'face':{
        'prefered_area': organo.face_df.mean(),
        'area_elasticity': 1,},
    'edge':{
        'line_tension': 1e-3,
        'is_active': 1
        },
    'vert':{
        'is_active': 1
        },
    }


organo.update_specs(specs)

energy = model.compute_energy(organo)
print(f'Computed enregy: {energy:.3f}')

def symmetric_energy(A_0, lbda, Nf, R_in, R_out):
    
    area = (R_out**2 - R_in**2) * sin(2*np.pi/Nf) / 2
    area_elasticity = 0.5 * (area - A_0)**2
    perimeter = 2*(R_out - R_in + (R_out + R_in) * sin(np.pi/Nf))
    line_tension = lbda * perimeter
    
    return area_elasticity + line_tension

area_0 = (R_out**2 - R_in**2) * sin(2*np.pi/Nf) / 2

areas, tensions = np.meshgrid(np.linspace(0.5, 1.5, 128)*area_0,
                              np.linspace(-1e-3, 1e-3, 128))

energies = symmetric_energy(areas, tensions,
                            Nf, R_in, R_out)

fig, ax = plt.subplots()
ax.pcolorfast([areas.min(), areas.max()],
              [tensions.min(), tensions.max()], energies)
ax.set(xlabel='Prefered Area',
       ylabel='Line tension')