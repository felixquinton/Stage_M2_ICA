#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 11:06:49 2018

@author: fquinton
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0, '../procedure_files')

#from tyssue.generation import generate_ring
from tyssue import PlanarGeometry
from tyssue.dynamics import effectors, factory
import imageProcessing as iP
import computingFunctions as cpF
import tyssueMissing
from tyssue.solvers.sheet_vertex_solver import Solver as solver
import time


path = '../images/seek4.png'
imgData = iP.dataImg(path)

#defining de organoid using the data we saved above
Nf = len(imgData.clockwiseCenters)
R_in = imgData.rIn/imgData.shape[1]
R_out = imgData.rOut/imgData.shape[1]
inners = imgData.inside
outers = imgData.outside
centers = imgData.clockwiseCenters

#compute the vertices of the mesh
inner_vs, outer_vs = cpF.get_bissecting_vertices(centers, inners, outers)

#initialising the mesh
organo = tyssueMissing.generate_ring(Nf, R_in, R_out)

# adjustement of the mesh to the vertices computed above.
organo.vert_df.loc[organo.apical_verts, organo.coords] = inner_vs
organo.vert_df.loc[organo.basal_verts, organo.coords] = outer_vs

PlanarGeometry.update_all(organo)

# Construction of the model
model = factory.model_factory(
    [effectors.FaceAreaElasticity,
     effectors.LineTension],
    effectors.FaceAreaElasticity)

# Model parameters or specifications
specs = {
    'face':{
        'is_alive': 1,
        'prefered_area': organo.face_df.area.mean(), #and there was an error here
        'area_elasticity': 1,},
    'edge':{
        'line_tension': 1e-3,
        'is_active': 1
        },
    'vert':{
        'is_active': 1
        },
    }

organo.update_specs(specs, reset=True)

Y = np.column_stack([organo.vert_df.x, organo.vert_df.y])

minimize_opt = {'options':{'gtol':0.001,
                           'ftol':0.01}}

def distance(L):   
    #initialising the mesh
    tmpOrgano = organo.copy()
    tmpOrgano.edge_df.line_tension = L
    solver.find_energy_min(tmpOrgano,
                             PlanarGeometry,
                            model,
                            minimize = minimize_opt)
    X = np.column_stack([tmpOrgano.vert_df.x, tmpOrgano.vert_df.y])
    D = np.sum(np.linalg.norm((X-Y), axis = 1))
    return D


class Swarm:
	
    def __init__(self,dimension, nb_iter, start_inertia, particule_trust, swarm_trust, criteria):
        self.vect = []
        self.dim = dimension
        self.iner = start_inertia
        self.p_t = particule_trust
        self.s_t = swarm_trust
        self.criteria = criteria
        self.inertia_decrease = (start_inertia-0.1)/float(nb_iter**2)
        pass

    def random_init(self, LB, UB, delta_t, length):
        self.LB = LB
        self.UB = UB
        self.length = length
        self.vect = np.array([LB + np.multiply((UB-LB),np.random.rand(self.dim)) for i in range(length)])
        self.speed = np.array([LB + np.multiply((UB-LB),np.random.rand(self.dim))/delta_t for i in range(length)])
        self.delta = delta_t
        self.swarm_min = np.array(self.vect[np.argmin([self.criteria(self.vect[i]) for i in range(length)])])
        self.particule_min = np.array([[self.vect[j][i] for i in range(self.dim)] for j in range(length)])
        pass

    def update(self,iteration):
        self.speed = 1/iteration*(self.iner*self.speed+
                                         self.p_t*np.random.rand(self.dim)*
                                         (self.particule_min-self.vect)/self.delta+
                                         self.s_t*np.random.rand(self.dim)*
                                         (self.swarm_min-self.vect)/self.delta)
        self.vect = self.vect + self.speed*self.delta
        craziness = np.random.rand(self.length)
        self.vect[craziness < 0.025] = self.LB + np.multiply((self.UB-self.LB),np.random.rand(self.dim))
        self.speed[craziness < 0.025] = self.LB + np.multiply((self.UB-self.LB),np.random.rand(self.dim))/self.delta
        updateMin = self.vect[np.argmin([self.criteria(self.vect[i]) for i in range(self.length)])]
        swarmMinCriteria = self.criteria(self.swarm_min)
        if self.criteria(updateMin) < swarmMinCriteria : self.swarm_min = updateMin
        for i in range(self.length):
            if self.criteria(self.vect[i]) < self.criteria(self.particule_min[i]) : 
                self.particule_min[i] = self.vect[i]
        self.iner -= self.inertia_decrease
        
 
nb_iter = 100
dimension = 4*Nf

start_inertia = 1.0
particule_trust = 2.0
swarm_trust = 2.0
LB = np.zeros(dimension)
UB = np.ones(dimension)*0.1
delta_t = 1.0
length = 50
print_mode = 1
       
def criteria(x):
    inBound = True
    for i in range(dimension):
        if not LB[i]<=x[i]<=UB[i] : inBound = False
    if inBound:return(distance(x))
    return(float('inf'))

start = time.clock()

swarm =  Swarm(dimension, nb_iter, start_inertia, particule_trust, swarm_trust, criteria)

swarm.random_init(LB, UB, delta_t, length)

#bar = progressbar.ProgressBar(redirect_stdout=False)
for i in range(nb_iter):
    print('update n° ',i)
    swarm.update(i+1)
    #bar.update(i*100.0/nb_iter)
    if print_mode==2 and dimension==2:swarm.plot()

print ('argmin f(x) = %s, objective value = %s'
       %(swarm.swarm_min,swarm.criteria(swarm.swarm_min)))

optL = swarm.swarm_min
elapsed = time.clock()-start

organo.edge_df.line_tension = optL
solver.find_energy_min(organo,
                            PlanarGeometry,
                            model,
                            minimize = minimize_opt)



# Plot of the mesh
fig, ax = plt.subplots()
ax.plot(*inners.T, lw=1, alpha=0.6, c='gray')
ax.plot(*outers.T, lw=1, alpha=0.6, c='gray')
fig, ax = tyssueMissing.quick_edge_drawMod(organo, ax=ax)

ax.set(aspect='equal');
print('Solving time =',elapsed)