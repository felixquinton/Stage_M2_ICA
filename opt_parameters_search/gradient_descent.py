#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:07:34 2018

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
from scipy import optimize as opt

brigthfield_path = '../images/Images/r01c01f03p55-ch3sk1fk1fl1.tiff'
dapi_path = '../images/Images/r01c01f03p55-ch1sk1fk1fl1.tiff'
imgData = iP.dataImage(brigthfield_path, dapi_path)
brigthfield_data = imgData.brightfield_processing()
dapi_data = imgData.dapi_processing()

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

# adjustement
organo.vert_df.loc[organo.apical_verts, organo.coords] = inner_vs[::-1]
organo.vert_df.loc[organo.basal_verts, organo.coords] = outer_vs[::-1]

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
        'ux': 0.,
        'uy': 0.,
        'uz': 0.,
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

def distance(P):   
    L, A = P[:4*Nf], P[4*Nf:]
    tmpOrgano = organo.copy()
    tmpOrgano.edge_df.line_tension = L
    tmpOrgano.face_df.prefered_area = A
    #start = time.clock()
    solver.find_energy_min(tmpOrgano,
                             PlanarGeometry,
                            model,
                            minimize = minimize_opt)
    X = np.column_stack([tmpOrgano.vert_df.x, tmpOrgano.vert_df.y])
    #elapsed = time.clock()-start
    #print('distance computation :',elapsed)
    D = np.sum(np.linalg.norm((X-Y), axis = 1))
    return D

#scipy approx_fprime is slower because it compute the partial derivatice twice for lateral edges 
def grad(P,D):
    h = np.array([10**(-6)]*len(P))
    hP = np.tile(P,(len(P),1)) + np.eye(len(P))*10**(-6)
    start = time.clock()
    Ldf = np.array([distance(i) for i in hP[:3*Nf]]) - np.full(3*Nf,D)
    Ldf = np.concatenate([Ldf,np.roll(Ldf[2*Nf:len(Ldf)],-1)])
    Adf = np.array([distance(i) for i in hP[4*Nf:]]) - np.full(Nf,D)
    df = np.concatenate((Ldf, Adf))
    elapsed = time.clock()-start
    print('gradient computation : ', round(elapsed,4))
    res = np.divide(df,h)
    return res

"""
Gradient descent : (much) faster than  simulated annealing.
"""
start = time.clock()
nonLateral = np.random.rand(int(len(organo.edge_df.line_tension)/2))*0.001
lateral = np.random.rand(int(len(organo.edge_df.line_tension)/4))*0.001
lateral = np.concatenate([lateral,np.roll(lateral,-1)])
L = np.concatenate([nonLateral, lateral])
#L contains line tensions. We need to ad equilibrium areas.
A = np.full(Nf, organo.face_df.area.mean())+np.random.rand(Nf)*0.01
P = np.concatenate((L,A))
D = distance(P)

previousStepSize = 10**6
cpt = 0
incumbent = 10**6
print(f'starting with {Nf} noyaux')
while previousStepSize > 10**(-5) and incumbent > 0:
    cpt += 1
    #L = np.maximum(L - 0.01/cpt * opt.approx_fprime(L, distance,10**(-6)),np.zeros(len(L)))    
    P = np.maximum(P - 0.01 / cpt * grad(P,D),np.zeros(len(P)))    
    D = distance(P)
    previousStepSize = abs(D - incumbent)
    print(f'Itération : {cpt-1} \n Distance : {round(D,5)}')
    print('________________________________________')
    incumbent = D
optP = P
elapsed = time.clock()-start

organo.edge_df.line_tension = optP[:4*Nf]
organo.face_df.prefered_area = optP[4*Nf:]
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