#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:07:34 2018

@author: fquinton
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:54:49 2018

@author: fquinton
"""

import numpy as np
import matplotlib.pyplot as plt
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

"""

PB : find_energy_min is too slow (between 1s and 8s)

"""
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


def grad(L,D):
    h = np.array([10**(-6)]*len(L))
    hL = [np.array(L) + np.array([(j==i)*10**(-6) for j in range(len(L))]) for i in range(len(L))]
    df = np.array([distance(i) for i in hL]) - np.array([D]*len(L))
    res = np.divide(df,h)
    return res
    
"""
Gradient descent : faster than  simulated annealing.
"""
start = time.clock()
L = np.random.rand(len(organo.edge_df.line_tension))*0.001
D = distance(L)

bnds = [(0, None)]*len(organo.edge_df.line_tension)

precision = np.ones(len(L))*10**(-5)
previousStepSize = 10**6
cpt = 0
incumbent = 10**6

while previousStepSize > 10**(-5) and incumbent > 0:
    cpt += 1
    L = np.maximum(L - 0.01/cpt * grad(L,D),np.zeros(len(L)))
    D = distance(L)
    previousStepSize = abs(D - incumbent)
    #print(f'Itération : {cpt-1} \n Distance : {D}')
    incumbent = D
optL = L
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