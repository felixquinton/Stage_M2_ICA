#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:54:49 2018

@author: fquinton
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../')

#from tyssue.generation import generate_ring
from tyssue import PlanarGeometry
from tyssue.draw.plt_draw import quick_edge_draw
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

# Plot of the mesh
fig, ax = plt.subplots()
ax.plot(*inners.T, lw=3, alpha=0.6)
ax.plot(*outers.T, lw=3, alpha=0.6)

fig, ax = quick_edge_draw(organo, ax=ax)

ax.set(aspect='equal');

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

#Simulated annealing - work but slow (~10min)

start = time.clock()
# Problems dimension
DIMENSION = 4*Nf
NB_TRANSITIONS = 100
ALPHA = .975

class Etat:
    def __init__(self, dim_etat):
        self.vecteur = []

    def init_aleatoire(self):
        nonLateral = np.random.randint(1,10,int(DIMENSION/2))*0.001
        lateral = np.random.randint(1,10,int(DIMENSION/4))*0.001
        lateral = np.concatenate([lateral,np.roll(lateral,-1)])
        self.vecteur = np.concatenate([nonLateral, lateral])
        pass

    def afficher(self):
        print(self.vecteur)
        return ""

    def generer_voisin(self):
        rand = int(np.floor(DIMENSION*np.random.rand()))
        self.old_index = {rand : self.vecteur[rand]}
        randDIR = 1 - 2*(np.random.rand()>0.5)
        self.vecteur[rand] = max(0,self.vecteur[rand] + 0.001*randDIR)
        pass
    
    def alternative_neigthbor(self):
        self.old_index = {}
        moving_indices = np.random.randint(0,2,DIMENSION)
        for indice in np.where(moving_indices == 1)[0]:
            self.old_index[indice] = self.vecteur[indice]
            randDIR = 1 - 2*(np.random.rand()>0.5)
            self.vecteur[indice] = max(0,self.vecteur[indice] + 0.001*randDIR)
        pass
   
    def mutation(self):
        self.old_index = {}
        for i in range(DIMENSION):self.old_index[i] = self.vecteur[i] 
        self.init_aleatoire()

    def come_back(self):
        for i in self.old_index:
            self.vecteur[i] = self.old_index[i]
        pass

    def calcul_critere(self):
        y = distance(self.vecteur)
        return y

class Recuit:
    def _accept(self, yi, yj, temperature):
        if yj < yi : return True
        else :
            proba = np.exp(-(yj-yi)/temperature)
            #print(np.exp(-(yj-yi)/temperature))
            tirage = np.random.rand()
            if tirage < proba : 
                self.probaAdd += 1
                return True
        return False

    def heat_up_loop(self):
        #Compute initial temperature

        temperature = 0.00001
        taux_acceptation = 0.0

        xi = Etat(DIMENSION)
        while taux_acceptation < 0.8:
            self.probaAdd = 0
            accept_count = 0
            temperature *= 1.1
            i = NB_TRANSITIONS
            while i > 0:
                i -= 1
                # Generate a point in space
                xi.init_aleatoire()
                yi = xi.calcul_critere()
                xi.generer_voisin()
                yj = xi.calcul_critere()
                if yi==yj:i += 1
                elif self._accept(yi, yj, temperature):
                    accept_count += 1
            taux_acceptation = float(accept_count) / NB_TRANSITIONS	
            #print(f"Temperature:{temperature} Acceptation rate:{taux_acceptation} Acceptés par proba :{self.probaAdd}")
        return temperature


    def cooling_loop(self, temperature_initiale):
        #Cooling process

        temperature = temperature_initiale

        xi = Etat(DIMENSION)
        xi.init_aleatoire()
        yi = xi.calcul_critere()
        while temperature > 0.0001 * temperature_initiale:
            self.probaAdd = 0
            i = NB_TRANSITIONS
            while i > 0:
                i -= 1
                if np.random.rand()<0.05 : xi.mutation()
                else : xi.generer_voisin()
                yj = xi.calcul_critere()
                if yj==0:
                    print('sortie précoce')
                    return xi.vecteur, yj
                if yi==yj:i += 1
                elif self._accept(yi, yj, temperature):
                    yi = yj
                else:
                    xi.come_back()
				    
            temperature *= ALPHA
		
            #print("Temperature : ", temperature, ", criteria value : ", yi, "Accepted by proba", self.probaAdd)
        return xi.vecteur, yi


if __name__ == "__main__":
    recuit = Recuit()
    print("Heatting.")
    temperature_initiale = recuit.heat_up_loop()
    print("Cooling")
    optL, optObj = recuit.cooling_loop(temperature_initiale)
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