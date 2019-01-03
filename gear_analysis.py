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
from scipy.signal import correlate


global sidelength
sidelength = 38.0

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
    meta = helpy.load_meta(prefix)
    meta.update(boundary = boundary)
    helpy.save_meta(prefix, meta)
    return boundary, first_tif

def correlation(a, b, cumulant = True, side = 'left'):
    '''
    left: a(t)*b(t+tau)
    right: a(t)*b(t-tau)
    '''
    
    if cumulant:
        a = a - a.mean()
        b = b - b.mean()
    d = correlate(a, b, mode = 'full')
    c = np.concatenate((np.cumsum(a*b), np.cumsum((a*b)[::-1])[::-1][1:]))
    l = len(a)
    cor = d/c
    if side == 'left':
        cor = cor[l-1:l-2-l//2:-1]
    elif side == 'right':
        cor = cor[l-1:l+l//2]
    return cor    

#def correlation(a, b, cumulant = False):
#    result = np.empty(0)
#    if cumulant:
#        a = a - a.mean()
#        b = b - b.mean()
#    for i in np.arange(len(a)//2):
#        cum, var = [0, 0]
#        for j in np.arange(len(a) - i):
#            cum += (a[j] * b[j + i])
#            var += (a[j] * b[j])
#        result = np.append(result, cum/var)
#    return result


def cc(a, b, cumulant = True, side = 'left'):
    '''
    left: a(t)*b(t+tau)
    right: a(t)*b(t-tau)
    '''
    if cumulant:
        a = a - a.mean()
        b = b - b.mean()
    cor = correlate(a, b, mode = 'full')
    n1 = np.cumsum(a**2)
    n2 = np.cumsum(b**2)
    n = n1 * n2
    norm = np.sqrt(np.concatenate((n, n[::-1][1:])))
    cor = cor / norm
    l = len(a)
    if side == 'left':
#        cor = cor[l-1:l-2-l//2:-1]
        cor = cor[l-1::-1]
    if side == 'right':
#        cor = cor[l-1:l+l//2]
        cor = cor[l-1::]
    return cor

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


 
def plot_r_density(r_distribution, total_frame = 6000.0, side = sidelength, 
                   r = 502.0, prefix = '', number = 300):
    r_sep = np.arange(r, 0 , -side)
    r_sep = np.append(r_sep, 0)
    ring_area = np.pi * np.array([r_sep[i]**2 - r_sep[i+1]**2 for i in range(len(r_sep) - 1)])
    n, bins = np.histogram(r_distribution,  r_sep[::-1])
    n = n * side**2 / total_frame   
    density = n / ring_area[::-1]
    figure, ax = plt.subplots()
    ax.bar(np.arange(len(density)), density, width=0.5, color = 'darkorange')
    ax.set_title('Density for each ring')
    ax.set_ylabel('Density')
    ax.set_xlabel('Ring number')
    ax.set_ylim([0,1])
    real = number / np.pi / 400
    ax.axhline(y = real, color = 'black', linestyle='dashed')
    ax.text(9.5, real + 0.05,  "%.2f" % real, color='black', fontweight='bold')
    #for i, v in enumerate(density):
    #    ax.text(i - 0.4, v , "%.2f" % v, color='blue', fontweight='bold')
    figure.savefig(prefix + '_Per_Ring.pdf')
    return


def plot_order_distribution(order_distribution, prefix = ''):
    figure, ax = plt.subplots()
    ax.hist(order_distribution, 100, range = [-1.5, 1.5], color = 'darkorange')
    ax.set_title('Order_Parameter_Distribution')
    figure.savefig(prefix + '_order_dist.pdf')
    return

def plot_vpar_distribution(vpar, prefix = ''):
    figure, ax = plt.subplots()
    ax.hist(vpar, 100, color = 'darkorange')
    ax.set_title('V_par_Distribution')
    figure.savefig(prefix + 'V_par_dist.pdf')
    return 
    
def plot_velocity_std(vstds, prefix = '', cutframe = 0):
    fig, ax = plt.subplots()
    ax.plot(vstds)
    ax.set_xlabel('frame')
    ax.set_ylabel(r'std of $V_{\theta}$')
    fig.savefig(prefix +'_veloctiy_std.pdf',dpi = 400) 
    return
    

def plot_distr(order_distr, vtheta_distr, vr_distr, vo_distr, prefix = '', cutframe = 0):
    figure, ax = plt.subplots(2,2, figsize = (15,10))
    ax[0,0].hist(order_distr, 30, color = 'r', log = True, alpha = 0.4, edgecolor = 'white', lw = 0.2)
    ax[0,0].set_title(r'$\Phi$ distribution')
    ax[0,0].axvline(np.nanmean(order_distr), color = 'black', ls = '--',lw = 2)
    
    ax[0,1].hist(vtheta_distr, 30, color = 'green', log = True, alpha = 0.4, edgecolor = 'white', lw = 0.2)
    ax[0,1].set_title(r'$V_{\theta}$ distribution')
    ax[0,1].axvline(np.nanmean(vtheta_distr), color = 'black', ls = '--', lw = 2)
    
    ax[1,0].hist(vr_distr, 30, color = 'b', log = True, alpha = 0.4, edgecolor = 'white', lw = 0.2)
    ax[1,0].set_title(r'$V_{r}$ distribution')
    ax[1,0].axvline(np.nanmean(vr_distr), color = 'black', ls = '--', lw = 2)
    
    
    ax[1,1].hist(vo_distr, 30, color = 'y', log = True, alpha = 0.4, edgecolor = 'white', lw = 0.2)
    ax[1,1].set_title(r' $V_{\omega}$ distribution')
    ax[1,1].axvline(np.nanmean(vo_distr), color = 'black', ls = '--', lw = 2)

    figure.savefig(prefix +'_distribution.pdf',dpi = 400)   
    return 


def plot_order(order, vring, vo, vr, prefix = '', cutframe = 0, smoothed = True):
    if smoothed:
        order = order[2: len(order)-2]
        vring = vring[2: len(vring)-2]
        vo = vo[40: len(vo) - 40]   
    figure, ax = plt.subplots(2,2, figsize = (15,10),sharex = True)
    ax[0,1].plot(np.arange(len(vring)), vring, color = 'r', linewidth = 0.8)
    ax[0,1].set_title(r'$V_{\theta}$ as function of t')
#    ax.set_xlabel('Frame Number')
    ax[0,1].set_ylabel(r'$V_{\theta}$')
    [y1_min, y1_max] = [0, 0.1] if np.nanmean(vring) > 0 else [-0.1, 0]
    ax[0,1].set_ylim([y1_min, y1_max])
#    figure.savefig(prefix + '_Velocity.pdf')
#    figure, ax = plt.subplots()
    ax[0,0].plot(np.arange(len(order)), order, color = 'green', linewidth = 0.8)
    ax[0,0].set_title(r'$\Phi$ as function of t')
    ax[0,0].set_ylabel(r'$\Phi$')
    [y2_min, y2_max] = [0, 1] if np.nanmean(vring) > 0 else [-1, 0]
    ax[0,0].set_ylim([y2_min, y2_max])
    ax[1,0].plot(np.arange(len(vr)), vr, color = 'yellow', lw = 0.8)
    ax[1,0].set_title(r'$V_{r}$ as function of t')
    ax[1,0].set_ylabel(r'$V_{r}$')
    ax[1,1].plot(np.arange(len(vo)), vo, color = 'blue', linewidth = 0.8)
    ax[1,1].set_title(r' $V_{\omega}$ as function of t')
    ax[1,1].set_ylabel(r' $V_{\omega}$')
    ax[1,1].set_ylim([-0.015,0.015])
    figure.savefig(prefix +'_velocity_order.pdf',dpi = 400)
    
    
    """
    figure, axr = plt.subplots(3, sharex = True)
    axr[0].plot(correlation(vring[cutframe:], vring[cutframe:], cumulant = True), 
       color = 'r', linewidth = 0.8)
    axr[0].set_title('Velocity Auto-Correlation')
    axr[0].set_ylim([0, 1.2])
    axr[1].plot(correlation(order[cutframe:], order[cutframe:], cumulant = True), 
       color = 'green', linewidth = 0.8)
    axr[1].set_title('Order Auto-Correlation')
    axr[1].set_ylim([0, 1.2])
    axr[2].plot(correlation(vo[cutframe:], vo[cutframe:], cumulant = True),
       color = 'b', linewidth = 0.8)
    axr[2].set_title('V_o Auto-Correlation')
    axr[2].set_ylim([0, 1.2])
    figure.savefig(prefix + '_Auto-Correlation.pdf')
    
    figure, ax = plt.subplots(2, sharex = True)
    ax[0].plot(correlation(vring[cutframe:], order[cutframe:], cumulant = True),
            color = 'r', linewidth = 0.8)
    ax[0].set_ylim(ymin = -0.2)
    ax[0].set_title('Vring Fluctuation Correlation Function')
    ax[1].plot(correlation(vo[cutframe:], order[cutframe:], cumulant = True),
            color = 'green', linewidth = 0.8)
    ax[1].set_ylim(ymin = -0.2)
    ax[1].set_title('Vorientation Fluctuation Correlation Function')
    figure.savefig(prefix + '_Fluctuation_Correlation.pdf')
    """
    return

def plot_stds(stds, prefix = ''):
    return



def plot_diff(diff, prefix = ''):
    figure, ax = plt.subplots()
    ax.plot(diff, linewidth = 0.8, color = 'blue')
    ax.set_title('+ and - difference')
    ax.set_ylabel('N+ - N-')
    figure.savefig(prefix + '_Difference.pdf')
    return


def gaussian_test():
    width = 3
    return width

def layer_analysis(prefix, cutframe = 0, layer_number = 1, grad = False, 
                   skip = 1, smoothed = False, fps = 2.5):
    meta = helpy.load_meta(prefix)
    boundary = meta.get('boundary')
    if boundary is None or boundary == [0.0]*3:
#    if True:
        boundary, path = find_boundary(prefix)
        meta['path_to_tiffs'] = path
        helpy.save_meta(prefix, meta)
    x0, y0, R = boundary
    

    data_path = prefix + '_ring_data.npy'
    if os.path.exists(data_path):
        v_data = np.load(data_path)
    if True:
        data = helpy.load_data(prefix)
        data['o'] = (data['o'] + np.pi)%(2 * np.pi)   # flip the detected orientation
        tracksets = helpy.load_tracksets(data, run_track_orient=True, run_repair = 'interp')
        track_prefix = {prefix: tracksets}
        v_data = velocity.compile_noise(track_prefix, width=(2.5,), cat = False, side = sidelength, fps = 2.5, 
                                   ring = True, x0= x0, y0 = y0, skip = skip, grad = grad)
        v_data = v_data[prefix]
        np.save(data_path, v_data)
    
    fdata = helpy.load_framesets(v_data)
    order, vsring, frame, number, difference, vo, vstds, vr = (np.empty(0) for k in range(8))
    r_density, ori_distr, order_distr, vpar, vtheta_distr, vr_distr, vo_distr= (np.empty(0) for k in range(7))
    for f, framedata in fdata.iteritems():
#        legal = data_filter(framedata, x0, y0, R - sidelength*layer_number, 
#                            R - sidelength*(layer_number - 1))
        mask = (framedata['r'] < R - sidelength * (layer_number - 1)) & (
                framedata['r'] > R - sidelength * layer_number)
#        length = len(legal[0])
        number = np.append(number,(sum(mask)))    # number in ring
        layer_data = framedata[mask]
        cen_orient = layer_data['corient']
        cor_orient = layer_data['o']
        cen_unit_vector = np.array([np.cos(cen_orient), np.sin(cen_orient)]).T
        cor_unit_vector = np.array([np.cos(cor_orient), np.sin(cor_orient)]).T
        ring_orient = np.cross(cen_unit_vector, cor_unit_vector)/np.sin(np.pi/4)
        clockwise = len(np.where(ring_orient > 0)[0])
        counter_clockwise = len(np.where(ring_orient < 0)[0])
        difference = np.append(difference, clockwise - counter_clockwise)   # n+ - n-
        vring = layer_data['vring']
#        frame = np.append(frame, f)
        order = np.append(order, (np.mean(ring_orient)))
        vsring = np.append(vsring, (np.mean(vring)))
        vstds = np.append(vstds, np.nanstd(vring))
        vr = np.append(vr, np.nanmean(layer_data['vpar']))
        '''
        # need more consideration
        '''
        vorient = layer_data['vo'] - vring / layer_data['r']
        vo = np.append(vo, (np.mean(vorient)))
        # from this line, deal with data out_of layer
        out_data = framedata[~mask]
        """
        v_p = np.append(v_p, out_data['v_p'])
        v_t = np.append(v_t, out_data['v_t'])
        v_o = np.append(v_o, out_data['vo'])
        """
        if f >= cutframe:
            r_density = np.concatenate((r_density, framedata['r']))
            #ori_distr = np.concatenate((ori_distr, layer_data['o'] % (2 * np.pi)))
            order_distr = np.concatenate((order_distr, ring_orient))
            vtheta_distr = np.concatenate((vtheta_distr, vring))
            vr_distr = np.concatenate((vr_distr, layer_data['vpar'] ))
            vo_distr = np.concatenate((vo_distr, vorient))
            vpar = np.concatenate((vpar, layer_data['vpar']))
#    plot_r_density(r_density, total_frame= len(frame) - cutframe ,
#                   side = sidelength, r = R, prefix = prefix)
    plot_velocity_std(vstds)
    plot_order(order = order, vring = vsring, vo = vo, vr = vr, prefix = prefix, smoothed= smoothed)
    plot_distr(order_distr, vtheta_distr, vr_distr, vo_distr, prefix)
    plot_diff(diff = difference, prefix = prefix)
#    plot_order_distribution(order_distribution= order_distr, prefix = prefix)
#    plot_vpar_distribution(vpar = vpar, prefix = prefix)
#    return  vsring, order, vo, order_distr, vpar
#    return order, vsring, r_density, vo
#    return frame, order, vsring, number, empty_ring

#if __name__ == '__main__':
#    prefix2 = '/Users/zhejun/Document/Result/0710_order/result'
#    layer_analysis(prefix2) 
    return
    
    
