#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 11:56:46 2018

@author: fquinton
"""


from tyssue import Sheet, config
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def quick_edge_drawMod(sheet, coords=['x', 'y'], ax=None, **draw_spec_kw):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    x, y = coords
    srce_x = sheet.upcast_srce(sheet.vert_df[x]).values
    srce_y = sheet.upcast_srce(sheet.vert_df[y]).values
    trgt_x = sheet.upcast_trgt(sheet.vert_df[x]).values
    trgt_y = sheet.upcast_trgt(sheet.vert_df[y]).values
    edge_line_tension = sheet.edge_df.line_tension.values
    
    lines_x, lines_y = np.zeros(2 * sheet.Ne), np.zeros(2 * sheet.Ne)
    lines_x[::2] = srce_x
    lines_x[1::2] = trgt_x
    lines_y[::2] = srce_y
    lines_y[1::2] = trgt_y
    lines_x = lines_x.reshape(int(len(lines_x)/2),2)
    lines_y = lines_y.reshape(int(len(lines_y)/2),2)
    # Trick from https://github.com/matplotlib/
    # matplotlib/blob/master/lib/matplotlib/tri/triplot.py#L65
    #lines_x = np.insert(lines_x, slice(None, None, 2), np.nan)
    #lines_y = np.insert(lines_y, slice(None, None, 2), np.nan)
    maxLinesTension = max(edge_line_tension)
    for i,j,l in zip(lines_x,lines_y,edge_line_tension):
        ax.plot(i, j,color=[1/maxLinesTension*l,1-1/maxLinesTension*l,0],lw=0.2+2.7/maxLinesTension*l)
    ax.set_aspect('equal')
    return fig, ax

class AnnularSheet(Sheet):
    """2D annular model of a cylinder-like monolayer.
    Provides syntactic sugar to access the apical, basal and
    lateral segments of the epithlium
    """
    def segment_index(self, segment, element):
        df = getattr(self, '{}_df'.format(element))
        return df[df['segment'] == segment].index

    @property
    def lateral_edges(self):
        return self.segment_index('lateral', 'edge')

    @property
    def apical_edges(self):
        return self.segment_index('apical', 'edge')

    @property
    def basal_edges(self):
        return self.segment_index('basal', 'edge')

    @property
    def apical_verts(self):
        return self.segment_index('apical', 'vert')

    @property
    def basal_verts(self):
        return self.segment_index('basal', 'vert')
    
def generate_ring(Nf, R_in, R_out, R_vit=None, apical='in'):
    specs = config.geometry.planar_spec()
    specs['settings'] = specs.get('settings', {})
    specs['settings']['R_in'] = R_in
    specs['settings']['R_out'] = R_out
    specs['settings']['R_vit'] = R_vit

    Ne = Nf * 4
    Nv = Nf * 2
    vert_df = pd.DataFrame(index=pd.Index(range(Nv), name='vert'),
                           columns=specs['vert'].keys(), dtype=float)
    edge_df = pd.DataFrame(index=pd.Index(range(Ne), name='edge'),
                           columns=specs['edge'].keys(), dtype=float)
    face_df = pd.DataFrame(index=pd.Index(range(Nf), name='face'),
                           columns=specs['face'].keys(), dtype=float)

    inner_edges = np.array(
        [[f0, v0, v1] for f0, v0, v1
         in zip(range(Nf), range(Nf), np.roll(range(Nf), -1))])

    outer_edges = np.zeros_like(inner_edges)
    outer_edges[:, 0] = inner_edges[:, 0]
    outer_edges[:, 1] = inner_edges[:, 2] + Nf
    outer_edges[:, 2] = inner_edges[:, 1] + Nf

    left_spokes = np.zeros_like(inner_edges)
    left_spokes[:, 0] = inner_edges[:, 0]
    left_spokes[:, 1] = outer_edges[:, 2]
    left_spokes[:, 2] = inner_edges[:, 1]

    right_spokes = np.zeros_like(inner_edges)
    right_spokes[:, 0] = inner_edges[:, 0]
    right_spokes[:, 1] = inner_edges[:, 2]
    right_spokes[:, 2] = outer_edges[:, 1]

    edges = np.concatenate([inner_edges, outer_edges,
                            left_spokes, right_spokes])

    edge_df[['face', 'srce', 'trgt']] = edges
    edge_df[['face', 'srce', 'trgt']] = edge_df[['face', 'srce', 'trgt']].astype(int)

    thetas = np.linspace(0, 2*np.pi, Nf,
                         endpoint=False)
    thetas += thetas[1] / 2

    thetas = thetas[::-1]
    # Setting vertices position (turning clockwise for correct orientation)
    vert_df.loc[range(Nf), 'x'] = R_in * np.cos(thetas)
    vert_df.loc[range(Nf), 'y'] = R_in * np.sin(thetas)
    vert_df.loc[range(Nf, 2*Nf), 'x'] = R_out * np.cos(thetas)
    vert_df.loc[range(Nf, 2*Nf), 'y'] = R_out * np.sin(thetas)

    vert_df['segment'] = 'basal'
    edge_df['segment'] = 'basal'
    if apical == 'out':
        edge_df.loc[range(Nf, 2*Nf), 'segment'] = 'apical'
        vert_df.loc[range(Nf, 2*Nf), 'segment'] = 'apical'
    elif apical == 'in':
        edge_df.loc[range(Nf), 'segment'] = 'apical'
        vert_df.loc[range(Nf), 'segment'] = 'apical'
    else:
        raise ValueError(f'apical argument not understood,'
                          'should be either "in" or "out", got {apical}')
    edge_df.loc[range(2*Nf, 4*Nf), 'segment'] = 'lateral'

    datasets = {'vert': vert_df,
                'edge': edge_df,
                'face': face_df}
    return AnnularSheet('ring', datasets,specs, coords=['x', 'y'])

def make_noisy_ring(radius, noise_amp, n_points=256):
    if not n_points == 256 : 
        noise = np.random.rand(n_points)*0.2
        thetas = np.linspace(0, 2*np.pi, n_points, endpoint=False)+noise
    else : thetas = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    noise = np.random.normal(scale=noise_amp, size=(n_points)).cumsum()
    noise *= np.sin(thetas/2)
    radius = radius + noise
    points = radius * np.vstack((np.cos(thetas), np.sin(thetas)))
    # recenter
    points -= points.mean(axis=1)[:, np.newaxis]

    # we transpose the result to have one coordinate pair per line
    return points.T