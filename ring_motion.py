#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:15:25 2017

@author: zhejun
"""


import velocity
import numpy as np
import helpy
import matplotlib.pyplot as plt
import os
import glob
from scipy.signal import correlate
import os
from scipy import signal

# Code used for analysis
###############################################################################

def correlation_function(a, b, cumulant = True, side = 'left', periodic = False):
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
        cor = cor[l-1:l//2 - 1 :-1]
    if side == 'right':
#        cor = cor[l-1:l+l//2]
        cor = cor[l-1:1+l//2]
    return cor

def auto_correlation_function_2d(matrix):
    for j in np.arange((matrix.shape[1])//2):
        for i in np.arange(matrix.shape[1] - j):
            corr = correlation_function(matrix[:,i], matrix[:,i+j])
            if i == 0:
                temp_matrix = corr
            else:
                temp_matrix = np.vstack((temp_matrix, corr))
        avg_corr = np.mean(temp_matrix, axis = 0)
        if j == 0:
            corr_2d = avg_corr
        else:
            corr_2d = np.vstack((corr_2d, avg_corr))
    return corr_2d

def auto2(matrix):
    return

def periodic_corr(a, cumulant = True):
    if cumulant:
        a = a - a.mean()
    cor = list()
    for roll in np.arange(0, -len(a)/2, -1):
        cor.append(sum(a*np.roll(a, roll)))
    return roll


def filter_shaker_freq(v, Q = 30.0, fps = 5.0, test = False):
    w0 = 2.0 / fps
    b, a = signal.iirnotch(w0, Q)
    fil_v = signal.filtfilt(b, a, v)
    if test:
        freq = np.fft.fft(v)
        pos = np.argsort(freq)[-3:-1:]
        freq[pos] = 0 
        fil_v = np.fft.ifft(freq)
    return fil_v

###############################################################################

def Find_Direct(PATH_TO_DATA, Result = 'result'):
    """
    find all directory under one directory, append 'result' 
    for one_density_analysis use
    """
    FILE_PATH = list()
    for direct_name in glob.glob(PATH_TO_DATA + '*/'):  # find all directoy under PATH_TO_DATA
        FILE_PATH.append(os.path.join(direct_name, Result))
    return FILE_PATH


def Group_Data_Save(PATH_TO_DATA, save_name = '', width = 5 , skip = 1,start = 0,
                    grad = False, filt = 5.0, fps = 5.0, file_name = "All_result", change_frequency = True):
    """
    save data from different path as a dictionary, use the path as the key
    """
    total_data = dict()
    prefixes = Find_Direct(PATH_TO_DATA)
    for prefix in prefixes:
#        one_data_path = prefix + '_single_density_'+ str(width) + '.npy'
        #if os.path.exists(one_data_path):
         #   total_data[prefix] = np.load(one_data_path).item()            
        #else:
         #   total_data[prefix] = one_density_analysis(prefix, width)
        if change_frequency:
            fps = 250.0/float(eval(prefix.split('/')[-2]))
        print(fps)
        total_data[prefix] = one_density_analysis(prefix, width = width , skip = skip, 
                  grad = grad, filt = filt, fps = fps, start = start)
        one_density_plots(prefix, width = width , skip = skip, grad = grad, 
                          filt = filt , fps = fps)
    np.save(os.path.join(PATH_TO_DATA, 
                         "All_result_skip_"+str(skip)+"_width_" + str(width)+ save_name + ".npy"), total_data)
    return total_data

def Group_Analysis(PATH_TO_DATA, file_name = "All_result.npy", filt = False, fps = 5.0):
    file_path = os.path.join(PATH_TO_DATA, file_name)
    if os.path.exists(file_path):
        total_data = np.load(file_path).item()
    else:
        total_data = Group_Data_Save(PATH_TO_DATA, file_name, filt = filt, fps = fps)
    return

                 
#One density load function
###############################################################################

def boundary(prefix):
    meta = helpy.load_meta(prefix)
    boundary = meta.get('boundary')
    if boundary is None or boundary == [0.0]*3:
        boundary, path = find_boundary(prefix)
        meta.update(boundary = boundary)
        meta['path_to_tiffs'] = path
        helpy.save_meta(prefix, meta)
    return boundary
        
    
def find_boundary(prefix):    
    file_path = prefix + '*.tif'
    tif_files = glob.glob(file_path)
    first_tif = tif_files[0]
    boundary = helpy.circle_click(first_tif)
    return boundary, first_tif


def load_one_density_data(prefix, sidelength = 38.0, width = 5, skip = 1, grad = False, start = 0, fps = 5.0, filt = False
                          rearange = True):
    x0, y0, R = boundary(prefix)
    data = helpy.load_data(prefix)
    data['o'] = (data['o'] + np.pi)%(2 * np.pi)   # flip the detected orientation
    tracksets = helpy.load_tracksets(data, run_track_orient=True, min_length = 20, run_repair = 'interp')
    track_prefix = {prefix: tracksets}
    v_data = velocity.compile_noise(track_prefix, width=(width,), cat = False, side = sidelength, fps = fps/float(skip), 
                               ring = True, x0= x0, y0 = y0, skip = skip, grad = grad, start = start)
    v_data = v_data[prefix]
    print(prefix)
    if filt:
        for key, values in v_data.items():
            values['vring'] = filter_shaker_freq(values['vring'], fps = fps) 
    if rearange:
        orient_dict = dict()
        temp = v_data.copy()
        for key, value in temp.items():
            orient_dict[value['corient'][0]] = key
        orient_sort = np.sort(orient_dict.keys())
        keys = np.sort(v_data.keys())
        for i in np.arange(len(v_data.keys())):
            v_data[keys[i]] = np.copy(temp[orient_dict[orient_sort[i]]])
        for key, value in v_data.items():
            for i in range(len(value['t'])):
                value['t'][i] = key
    fdata = helpy.load_framesets(v_data)
    np.save((prefix +'_v_data_'+str(width)+'.npy'), v_data)
    np.save((prefix +'_frame_data_'+str(width)+'.npy'), fdata)
    return fdata, v_data

#In dividual Plot function
###############################################################################
def plot_vcross(correlation, prefix, width):
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(correlation)), correlation, 'bo', np.arange(len(correlation)), correlation, 'b--')
    ax.set_xlabel('Position Difference j')
    ax.set_ylabel('Correlation')
    #ax.set_title('Correlation average by time')
    fig.savefig(prefix + '_vcorrelation_' + str(width) + '.jpeg')
    return

def plot_std(std, prefix, width):
    fig, ax = plt.subplots()
    ax.plot(std, 'r--' )
    ax.set_xlabel('frame')
    ax.set_ylabel('std in each frame')
    fig.savefig(prefix + "_std_" + str(width) + ".pdf")
    return

def plot_std_zoom(std, prefix, width, zoom = 200):
    fig, ax = plt.subplots()
    ax.plot(std[:zoom], 'r--' )
    ax.set_xlabel('frame')
    ax.set_ylabel('std in each frame')
    fig.savefig(prefix + "_std_zoom" + str(width) + ".pdf")
    return


def plot_velocity_map(matrix, prefix, name, width):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues, alpha=0.8)
    fig = plt.gcf()
    fig.set_size_inches(8,20)
    cbar = fig.colorbar(heatmap, shrink = 1.0,aspect= 40)
    cbar.ax.tick_params(labelsize=12)
    ax.set_frame_on(False)
    ax.grid(False)
    ax = plt.gca()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
    #ax.imshow(matrix, origin = "lower", cmap = "Blues", interpolation = "nearest")
    ax.xaxis.set_ticks(np.arange(0, 30, 5))
    ax.yaxis.set_ticks(np.arange(0,550, 50))
    ax.set_title("Velocity Colormap", fontsize = 24)
    ax.set_xlabel("Particle ID", fontsize = 24)
    ax.set_ylabel("Time(frame)", fontsize = 24)
    fig.savefig(prefix + name + "_colormap_" + str(width) + ".jpeg")
    return

def plot_2d_correlation(matrix, prefix, width):
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(matrix, cmap=plt.cm.Blues, alpha=0.8)
    fig = plt.gcf()
    fig.set_size_inches(20,8)
    ax.set_frame_on(False)
    ax.grid(False)
    ax = plt.gca()
    fig.savefig(prefix + "_correlation_map_" + str(width) + ".pdf")
    return

def plot_individual_velocity(v, prefix, id_number, width):
    fig, ax = plt.subplots()
    ax.plot(v, 'b')
    ax.set_title("velocity vs frame")
    fig.savefig(prefix + "_velocity_" + str(id_number) + str(width) + ".pdf")
    return

def plot_time_corr(cor, prefix, number, width):
    import matplotlib as mpl
    #mpl.rc('text', usetex = True)
    fig, ax = plt.subplots()
    argmin = np.argmin(cor)
    ax.plot(cor[:argmin*12], color = 'blue')
    ax.set_ylabel("Time Correlation")
    ax.set_xlabel("$\Delta t$")
    #ax.set_title("Time Correlation Averaged by Particles")
    fig.savefig(prefix + "_time_corr_" + str(number) + "_" + str(width) + ".jpeg")
    return



#Group Plot Function
###############################################################################

def diff_quantity_plot(file_name, save_name, quantity ,width = 5):
    direct_name = os.path.dirname(file_name)
    cor_den = np.load(file_name).item()
    fig, ax = plt.subplots()
    keys = cor_den.keys()
    number_keys = [eval(key.split('/')[-2]) for key in keys]
    sort_index = np.argsort(number_keys)
    sort_keys = np.array(keys)[sort_index]
    for key in sort_keys:
        ax.scatter(np.arange(20), cor_den[key]['time_correlation'][:20], linewidth = 2.2 , label = key.split('/')[-2])
    ax.axhline(y = 0, color = 'black', linestyle = '--')
    ax.legend(fancybox=True)
    ax.set_title("time correlation of different " + quantity)
    #ax.set_ylim([-0.2,0.4])
    fig.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_time_correlation.pdf")))
    fig2, ax2 = plt.subplots()
    for key in sort_keys:
        ax2.plot(cor_den[key]['correlation'], 'o--', lw = 2.5, label = key.split('/')[-2])
    ax2.axhline(y = 0, color = 'black', linestyle = '--')
    ax2.legend(fancybox=True)
    #ax2.set_yscale('log')
    #ax2.set_xscale('log')
    ax2.set_title("Particle ID correlation of different " + quantity)
    fig2.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_velocity_correlation.pdf")))
    fig3, ax3 = plt.subplots()
    for key in sort_keys:
        ax3.scatter(key.split('/')[-2] ,np.abs(cor_den[key]['v_average']) , alpha = 0.6, label = key.split('/')[-2])
    ax3.set_title("Average Velocity of Different " + quantity)
    ax3.legend(fancybox=True)
    fig3.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_average_velocity.pdf")))
    fig4, ax4= plt.subplots()
    for key in sort_keys:
        ax4.scatter(key.split('/')[-2] ,cor_den[key]['v_std'], alpha = 0.6, label = key.split('/')[-2])
    ax4.set_title("Std of Velocity of Different " + quantity)
    ax4.legend(fancybox=True)
    fig4.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_velocity_std.pdf")))    
    return


# One density Analysis and Result Plot
###############################################################################
def vdata_analysis(prefix, width = 5, skip = 1, start = 0, grad = False, filt = False, fps = 5.0):
    # get vdata
    v_path = prefix + '_v_data_'+ str(width) + '.npy'
    if False:
    #if os.path.exists(v_path):
        vdata = np.load(v_path).item()
    else:
        vdata = load_one_density_data(prefix, width = width, skip = skip, 
                                      grad = grad, start = start, filt = filt, fps = fps)[1]
    #vstack vmatrix, axis = 0 is particle id, 
    for key, value in vdata.items():
        if key == 0:
            length = len(value['vring'])
            v_matrix = value['vring']
        else:
            if len(value['vring']) == length:
                v_matrix = np.vstack((v_matrix, value['vring']))
    #calculate time correlation 
    for i in np.arange(v_matrix.shape[0]):
        if i == 0:
            vcor = correlation_function(v_matrix[i], v_matrix[i])
        else:
            vcor = np.vstack((vcor, correlation_function(v_matrix[i], v_matrix[i])))
    
    vcor_average = np.mean(vcor, axis = 0)    
    return vcor_average, v_matrix.T


def one_density_analysis(prefix, width = 5, skip = 1, bulk = False, grad = False, 
                         start = 0, filt = False, fps = 5.0):
    x0, y0, R = boundary(prefix)
    frame_path = prefix + '_frame_data_'+ str(width) + '.npy'
    if False:
    #if os.path.exists(frame_path):
        fdata = np.load(frame_path).item()
    else:
        fdata = load_one_density_data(prefix, width = width, skip = skip, grad = grad, filt = filt, fps = fps)[0]    
    vring_dist = np.empty(0)   # vring dist
    op_t = np.empty(0)  # op for each frame
    vr_time_op_dist= np.empty(0)  # vring * sign of op
    op_dist = np.empty(0)   # op distribution  in ring
    vring_t = np.empty(0)  # v for each frame in ring
    vstd_t = np.empty(0) # std of v for each frame in ring
    ps, ng, diff = list(), list(), list()
    vp_t = np.empty(0) # v_p in bulk
    vp_dist = np.empty(0) #vp in bulk
    vt_dist = np.empty(0)  # vt in bulk
    v_par_matrix = np.empty(0) # vpar matrix
    vr_time_op_t = np.empty(0)
    for f, framedata in fdata.iteritems():
        mask = ( framedata['r'] < R ) & (framedata['r'] > R - 55)
        layer_data = framedata[mask]  # in layer
        cen_orient = layer_data['corient']
        cor_orient = layer_data['o']
        cen_unit_vector = np.array([np.cos(cen_orient), np.sin(cen_orient)]).T
        cor_unit_vector = np.array([np.cos(cor_orient), np.sin(cor_orient)]).T
        ring_orient = - np.cross(cen_unit_vector, cor_unit_vector)/np.sin(np.pi/4)
        op_dist = np.append(op_dist, ring_orient)
        if f == 0:
            v_matrix = layer_data['vring']
            v_err_matrix = layer_data['vring'] - layer_data['vring'].mean()
            v_corr = correlation_function(layer_data['vring'], layer_data['vring'])
            v_par_matrix = layer_data['vpar']
            number = sum(mask)
            
        else:
            v_matrix = np.vstack((v_matrix, layer_data['vring']))
            v_err_matrix = np.vstack((v_err_matrix, layer_data['vring'] - layer_data['vring'].mean()))
            v_corr = np.vstack((v_corr, correlation_function(layer_data['vring'], layer_data['vring'])))
            v_par_matrix = np.vstack((v_par_matrix, layer_data['vpar']))
        vring_dist = np.append(vring_dist, layer_data['vring'])
        vstd_t = np.append(vstd_t, layer_data['vring'].std())
        vr_time_op_dist = np.append(vr_time_op_dist, layer_data['vring'] * np.sign(ring_orient))
        vr_time_op_t = np.append(vr_time_op_t, np.nanmean(layer_data['vring'] * np.sign(ring_orient)))
        op_t = np.append(op_t, np.nanmean(ring_orient))
        vring_t = np.append(vring_t, np.nanmean(layer_data['vring']))
        ps.append(sum(ring_orient > 0))
        ng.append(sum(ring_orient < 0))
        diff.append(sum(ring_orient > 0) - sum(ring_orient < 0) )
        # outring data, interaction into consideration
        if bulk:
            outdata = framedata[~mask]
            vp_dist = np.append(vp_dist, outdata['v_p'])
            vt_dist = np.append(vt_dist, outdata['v_t'])
            vp_t = np.append(vp_t, np.nanmean(outdata['v_p']))
    number = np.asarray(number)
    correlation = np.mean(v_corr, axis = 0)
    all_data = ['vring_dist', 'op_t', 'vr_time_op_dist', 'op_dist', 'vring_t',
                'vstd_t', 'vr_time_op_t', 'correlation', 'v_matrix', 
                'v_err_matrix', 'number']
    ring_data_result = dict()
    for name in all_data:
        ring_data_result[name] = eval(name)
    time_correlation = vdata_analysis(prefix, width = width, skip = skip, 
                                      start = start, filt = filt, grad = grad, 
                                      fps = fps)[0]
    ring_data_result['time_correlation'] = time_correlation
    ring_data_result['v_average'] = np.nanmean(v_matrix)
    ring_data_result['v_std'] = np.nanstd(v_matrix)
    ring_data_result['v_par_std'] = np.nanstd(v_par_matrix)
    np.save(prefix + '_single_density_' + str(width) + '.npy',ring_data_result)
    return ring_data_result


def one_density_plots(prefix, width = 5, skip= 1, grad = False, filt = False, fps = 5.0):
    file_name = prefix + '_single_density_' + str(width) + '.npy'
    if False:
    #if os.path.exists(prefix + '_single_density_' + str(width) + '.npy'):
        ring_data_result = np.load(prefix + '_single_density_' + str(width) + '.npy').item()
    else:
        ring_data_result = one_density_analysis(prefix, width = width, skip = skip, grad = grad, filt = filt, fps = fps)
    vcor, vmatrix = vdata_analysis(prefix, width = width, skip = skip, grad = grad, filt = filt, fps = fps)
    cor2d = auto_correlation_function_2d(vmatrix)
    plot_2d_correlation(cor2d, prefix, width)
    plot_time_corr(vcor, prefix, ring_data_result['number'], width)
    plot_vcross(ring_data_result['correlation'], prefix, width)
    plot_std(ring_data_result['vstd_t'], prefix, width)
    plot_std_zoom(ring_data_result['vstd_t'], prefix, width)
    plot_velocity_map(ring_data_result['v_matrix'][:500], prefix, "_v", width)
    plot_velocity_map(ring_data_result['v_err_matrix'][:500], prefix, "_verror", width)
    plot_individual_velocity(ring_data_result['v_matrix'][:,1], prefix, 1, width)
    return 
    
    
    
    
    

    