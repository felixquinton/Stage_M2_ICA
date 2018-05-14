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
import computingFunctions as cpF
import tyssueMissing
from tyssue.solvers.sheet_vertex_solver import Solver as solver
import time
from scipy import optimize as opt
from tyssue.dynamics import units


class AnnularGeometry(PlanarGeometry):

    @classmethod
    def update_all(cls, eptm):
        PlanarGeometry.update_all(eptm)
        cls.update_lumen_volume(eptm)

    @staticmethod
    def update_lumen_volume(eptm):
        srce_pos = eptm.upcast_srce(eptm.vert_df[['x', 'y']]).loc[eptm.apical_edges]
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[['x', 'y']]).loc[eptm.apical_edges]
        apical_edge_pos = (srce_pos + trgt_pos)/2
        apical_edge_coords = eptm.edge_df.loc[eptm.apical_edges,
                                              ['dx', 'dy']]
        eptm.settings['lumen_volume'] = (
            - apical_edge_pos['x'] * apical_edge_coords['dy']
            + apical_edge_pos['y'] * apical_edge_coords['dx']).values.sum()


class LumenElasticity(effectors.AbstractEffector):
    '''

    .. math:: \frac{K_Y}{2}(A_{\mathrm{lumen}} - A_{0,\mathrm{lumen}})^2

    '''
    dimensions = units.area_elasticity
    label = 'Lumen volume constraint'
    magnitude = 'lumen_elasticity'
    element = 'settings'
    spatial_ref = 'lumen_prefered_vol', units.area

    specs = {
        'settings': {
            'lumen_elasticity',
            'lumen_prefered_vol',
            'lumen_volume'
            }
        }

    @staticmethod
    def energy(eptm):


        Ky = eptm.settings['lumen_elasticity']
        V0 = eptm.settings['lumen_prefered_vol']
        Vy = eptm.settings['lumen_volume']
        return np.array([Ky * (Vy - V0)**2 / 2,])

    @staticmethod
    def gradient(eptm):
        Ky = eptm.settings['lumen_elasticity']
        V0 = eptm.settings['lumen_prefered_vol']
        Vy = eptm.settings['lumen_volume']
        grad_srce, grad_trgt = lumen_area_grad(eptm)
        return (Ky*(Vy - V0) * grad_srce,
                Ky*(Vy - V0) * grad_trgt)


def lumen_area_grad(eptm):
    apical_pos = eptm.vert_df[['x', 'y']].copy()
    apical_pos.loc[eptm.apical_verts] = 0
    srce_pos = eptm.upcast_srce(apical_pos)
    trgt_pos = eptm.upcast_trgt(apical_pos)
    grad_srce = srce_pos.copy()
    grad_srce.columns = ['gx', 'gy']
    grad_trgt = grad_srce.copy()
    grad_srce['gx'] = trgt_pos['y']
    grad_srce['gy'] = -trgt_pos['x']
    grad_trgt['gx'] = -srce_pos['y']
    grad_trgt['gy'] = srce_pos['x']
    # minus sign due to the backward orientation
    return -grad_srce, -grad_trgt




#defining de organoid using the data we saved above
Nf = 6
R_in = 0.5
R_out = 1

#initialising the mesh
organo = tyssueMissing.generate_ring(Nf, R_in, R_out)

# adjustement
AnnularGeometry.update_all(organo)


# Construction of the model
model = factory.model_factory(
    [LumenElasticity,
     effectors.FaceAreaElasticity,
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
    'settings': {
        'lumen_elasticity': 10,
        'lumen_prefered_vol': organo.settings['lumen_volume'],
        'lumen_volume': organo.settings['lumen_volume']
        }
    }

minimize_opt = {'options':{'gtol':0.00001,
                           'ftol':0.0001}}

organo.update_specs(specs, reset=True)

basal = np.random.rand(Nf)*0.002
apical = np.random.rand(Nf)*0.004
lateral = np.random.rand(Nf)*0.001
lateral = np.concatenate([lateral,np.roll(lateral,-1)])
startL = np.concatenate([apical, basal, lateral])

startA = np.full(Nf, organo.face_df.area.mean())+np.random.rand(Nf)*0.01

organo.edge_df.line_tension = startL
organo.face_df.prefered_area = startA

solver.find_energy_min(organo,
                       AnnularGeometry,
                       model,
                       minimize = minimize_opt)

energy = model.compute_energy(organo)
print(f'Computed enregy: {energy:.3f}')
# Plot of the mesh
fig, ax = plt.subplots()

fig, ax = tyssueMissing.quick_edge_drawMod(organo, ax=ax)

plt.title('With true parameters')
Y = np.column_stack([organo.vert_df.x, organo.vert_df.y])



def distance(P):
    L, A = P[:4*Nf], P[4*Nf:]
    tmpOrgano = organo.copy()
    tmpOrgano.edge_df.line_tension = L
    tmpOrgano.face_df.prefered_area = A
    #start = time.clock()
    solver.find_energy_min(tmpOrgano,
                            AnnularGeometry,
                            model,
                            minimize = minimize_opt)
    X = np.column_stack([tmpOrgano.vert_df.x, tmpOrgano.vert_df.y])
    #elapsed = time.clock()-start
    #print('distance computation :',elapsed)
    # Plot of the mesh
    N = np.linalg.norm((X-Y), axis = 1)
#    D = np.sum(N[N>0.01])
#    D = np.max(N)
    D = np.sum(N)
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
nonLateral = np.maximum(np.zeros(2*Nf),startL[:2*Nf]+(np.random.rand(2*Nf)*0.001-0.0005))
lateral = np.maximum(np.zeros(Nf),startL[2*Nf:3*Nf]+(np.random.rand(Nf)*0.001-0.0005))
lateral = np.concatenate([lateral,np.roll(lateral,-1)])
L = np.concatenate([nonLateral, lateral])
#L contains line tensions. We need to ad equilibrium areas.
A =np.maximum(np.zeros(Nf),startA +np.random.rand(Nf)*0.01)
#P = np.concatenate((L,A))
P = np.concatenate((startL,startA))
D = distance(P)

previousStepSize = 10**6
cpt = 0
incumbent = D

while previousStepSize > 10**(-5) and incumbent > 0.001:
    cpt += 1
    #L = np.maximum(L - 0.01/cpt * opt.approx_fprime(L, distance,10**(-6)),np.zeros(len(L)))
    P = np.maximum(P - 0.01 / cpt * grad(P,D),np.zeros(len(P)))
    D = distance(P)
    previousStepSize = abs(D - incumbent)
    print(f'Itération : {cpt-1} \n Distance : {round(D,5)}')
    print('________________________________________')
    incumbent = D
optL, optA = P[:4*Nf], P[4*Nf:]
elapsed = time.clock()-start

organo.edge_df.line_tension = optL
organo.face_df.prefered_area = optA
solver.find_energy_min(organo,
                       AnnularGeometry,
                       model,
                       minimize = minimize_opt)


energy = model.compute_energy(organo)
print(f'Computed enregy: {energy:.3f}')
# Plot of the mesh
fig, ax = plt.subplots()
fig, ax = tyssueMissing.quick_edge_drawMod(organo, ax=ax)

ax.set(aspect='equal')
plt.title('With random starting parameters')
print('Solving time =',elapsed)
