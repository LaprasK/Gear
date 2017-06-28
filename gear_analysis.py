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
from orientation import get_angles
#from tracks import find_tracks
import velocity


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
    

def layer_analysis(prefix):
    data = helpy.load_data(prefix)
    meta = helpy.load_meta(prefix)
    boundary = meta.get('boundary')
    if boundary is None or boundary == [0.0]*3:
        boundary, path = find_boundary(prefix)
        meta.update(boundary = boundary)
        meta['path_to_tiffs'] = path
        helpy.save_meta(prefix, meta)
    x0, y0, R = boundary
    data['o'] = (data['o'] + np.pi)%(2 * np.pi)   # flip the detected orientation
    tracksets = helpy.load_tracksets(data, run_track_orient=True)
    track_prefix = {prefix: tracksets}
    v_data = velocity.compile_noise(track_prefix, cat = True, side = 36, fps = 2.5, 
                               ring = True, x0= x0, y0 = y0)
    fdata = helpy.load_framesets(v_data)
    order = list()
    vsring = list()
    frame = list()
    for f, framedata in fdata.iteritems():
        legal = data_filter(framedata, x0, y0, R - 36, R)
        legal_data = framedata[legal]
        cen_orient = legal_data['corient']
        cor_orient = legal_data['o']
        cen_unit_vector = np.array([np.cos(cen_orient), np.sin(cen_orient)]).T
        cor_unit_vector = np.array([np.cos(cor_orient), np.sin(cor_orient)]).T
        ring_orient = np.cross(cen_unit_vector, cor_unit_vector)
        order_par = np.nanmean(ring_orient)
        vring = legal_data['vring']
        vsum = np.nanmean(vring)
        frame.append(f)
        order.append(order_par)
        vsring.append(vsum)
    return frame, order, vsring
    
#    posdata = data_filter(pdata, x0, y0, R - 40, R + 5)
#    cordata = data_filter(cdata, x0, y0, R - 55, R + 10)
#    fpdata = helpy.load_framesets(posdata)
#    fcdata = helpy.load_framesets(cordata)
#    pftrees = {f: KDTree(helpy.consecutive_fields_view(fpset, 'xy'), leafsize = 50)
#                for f, fpset in fpdata.iteritems()}
#    cftrees = {f: KDTree(helpy.consecutive_fields_view(fcset, 'xy'), leafsize = 50)
#                for f, fcset in fcdata.iteritems()}
#    odata, omask = get_angles(posdata, cordata, fpdata, fcdata, cftrees, nc = 3,
#                              rc = 19, drc = 7, ang = 90, dang = 20)
#    trackids = None
#    data = helpy.initialize_tdata(posdata, trackids, odata['orient'])
#    fdata = helpy.load_framesets(data)
#    order = list()
#    frame = list()
#    for f, framedata in fdata.iteritems():
#        fpos = helpy.consecutive_fields_view(framedata, 'xy')
#        position = fpos - [x0, y0]
#        cororient = framedata['o']
#        cen_unv = normalize(position)
#        orient = (cororient + np.pi) % (2 * np.pi)
#        cor_unv = np.array([np.cos(orient), np.sin(orient)]).T
#        cross_product = np.array([np.cross(a, b) for a, b in zip(cen_unv, cor_unv)])
#        order.append(np.sum(cross_product))
#        frame.append(f)   
#    plt.plot(frame, order)
#    return 


        
                          