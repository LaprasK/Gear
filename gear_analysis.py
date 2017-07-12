#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 14:29:14 2017

@author: zhejun
"""

from __future__ import division

import glob
import numpy as np
import correlation as corr
import helpy
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree as KDTree
#from orientation import get_angles
#from tracks import find_tracks
import velocity
import correlation as corr
import os.path


def data_filter(data, x0 , y0 , inner, outer):
    x = data['x']
    y = data['y']
    position = [x - x0, y - y0]
    distance = np.hypot(*position)
    legal = np.where((distance < outer) & (distance > inner))
    return legal

def normalize(vector):
    dis = np.hypot(*vector.T)
    uv = np.array([vector[i]/dis[i] for i in range(len(vector))])
    return uv


def find_boundary(prefix):    
    file_path = prefix + '*.tif'
    tif_files = glob.glob(file_path)
    first_tif = tif_files[0]
    boundary = helpy.circle_click(first_tif)
    return boundary, first_tif

def c3(a, b, cumulant = True):
    if cumulant:
        a = a - a.mean()
        b = b - b.mean()
    from scipy.signal import correlate
    d = correlate(a, b, mode = 'full')
    c = np.concatenate((np.cumsum(a*b), np.cumsum((a*b)[::-1])[::-1][1:]))
    return d/c    

def correlation(a, b, cumulant = False):
    result = np.empty(0)
    if cumulant:
        a = a - a.mean()
        b = b - b.mean()
    for i in np.arange(len(a)//2):
        cum, var = [0, 0]
        for j in np.arange(len(a) - i):
            cum += (a[j] * b[j + i])
            var += (a[j] * b[j])
        result = np.append(result, cum/var)
    return result


def c2(a, b, cumulant = True):
    result = np.empty(0)
    if cumulant:
        a = a - a.mean()
        b = b - b.mean()
    for i in np.arange(len(a)/2 + 1):
        cum, var = [0, 0]
        for j in np.arange(len(a) - i):
            cum += (a[j] * b[j + i])
        result = np.append(result, cum/(len(a) - i) )
    return result
 
def plot_r_density(r_distribution, total_frame = 5999.0, side = 36.0, 
                   r = 502.0, prefix = ''):
    ring_number = np.ceil(r/float(side))
    n, bins = np.histogram(r_distribution, ring_number)
    n = n * side**2 / total_frame
    r_sep = np.arange(r, 0 , -side)
    r_sep = np.append(r_sep, 0)
    ring_area = np.pi * np.array([r_sep[i]**2 - r_sep[i+1]**2 for i in range(len(r_sep) - 1)])
    density = n / ring_area[::-1]
    figure, ax = plt.subplots()
    ax.bar(np.arange(1, len(density)+1), density, width=0.5, color = 'green')
    ax.set_title('Density for each ring')
    ax.set_ylabel('Density')
    ax.set_xlabel('Ring number')
    figure.savefig(prefix + '_Per_Ring.pdf')
    return

def plot_order(order, vring, prefix):
    figure, ax = plt.subplots()
    return
    

def layer_analysis(prefix):

    meta = helpy.load_meta(prefix)
    boundary = meta.get('boundary')
    if boundary is None or boundary == [0.0]*3:
        boundary, path = find_boundary(prefix)
        meta.update(boundary = boundary)
        meta['path_to_tiffs'] = path
        helpy.save_meta(prefix, meta)
    x0, y0, R = boundary
    

    data_path = prefix + '_ring_data.npy'
    if os.path.exists(data_path):
        v_data = np.load(data_path)
    else:
        data = helpy.load_data(prefix)
        data['o'] = (data['o'] + np.pi)%(2 * np.pi)   # flip the detected orientation
        tracksets = helpy.load_tracksets(data, run_track_orient=True, run_repair = 'interp')
        track_prefix = {prefix: tracksets}
        v_data = velocity.compile_noise(track_prefix, width=(20,), cat = True, side = 36, fps = 2.5, 
                                   ring = True, x0= x0, y0 = y0)
        np.save(data_path, v_data)
    
    fdata = helpy.load_framesets(v_data)
    order, vsring, frame, number, difference, vo = (list() for k in range(6))
    r_density, ori_distr, order_distr= (np.empty(0) for k in range(3))
    for f, framedata in fdata.iteritems():
        legal = data_filter(framedata, x0, y0, R - 40, R)
        length = len(legal[0])
        number.append(length)    # number in ring
        legal_data = framedata[legal]
        cen_orient = legal_data['corient']
        cor_orient = legal_data['o']
        vorient = legal_data['vo']
        cen_unit_vector = np.array([np.cos(cen_orient), np.sin(cen_orient)]).T
        cor_unit_vector = np.array([np.cos(cor_orient), np.sin(cor_orient)]).T
        ring_orient = - np.cross(cen_unit_vector, cor_unit_vector)
        clockwise = len(np.where(ring_orient > 0)[0])
        counter_clockwise = len(np.where(ring_orient < 0)[0])
        difference.append(clockwise - counter_clockwise)   # n+ - n-
        vring = legal_data['vring']
        frame.append(f)
        order.append(np.mean(ring_orient)/np.sin(np.pi/4))
        vsring.append(np.mean(vring))
        vo.append(np.mean(vorient))
        if f > 4000:
            r_density = np.concatenate((r_density, framedata['r']))
            ori_distr = np.concatenate((ori_distr, legal_data['o'] % (2 * np.pi)))
            order_distr = np.concatenate((order_distr, ring_orient))        
    plot_plot_r_density(r_density, side = 36.0, r = R, prefix = prefix)
    return 
#    return order, vsring, r_density, vo
#    return frame, order, vsring, number, empty_ring

def main(prefix):
    return

    
    
