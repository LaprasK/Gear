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
from scipy.signal import find_peaks
from scipy import signal
from scipy import interpolate
from collections import defaultdict
import matplotlib.cm as mplcm
import matplotlib.colors as colors


sep_k = 1500

# Code used for analysis
###############################################################################

def angular_distr(angles, bins, fluct = False):
    delta_density, ranges = np.histogram(angles, bins, [0, np.pi*2])
    if fluct == True:
        delta_density = delta_density - len(angles)/float(sep_k)
    sq = np.fft.fft(delta_density)
    sq2 = sq * np.conj(sq)
    dist = np.fft.ifft(sq2)
    return dist/dist[0]



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


def vector_correlation(a, b, limit = 3500,cumulant = True, side = 'left'):
    '''
    a, b matrix, vector along 0 axis
    '''
    dimen = a.shape[0]
    if cumulant :
        a = a - a.mean(axis = 0)
        b = b - b.mean(axis = 0)
    corr = list()
    for j in np.arange(limit):
        value = np.sum(a[:dimen - j]*b[j:])
        norm = np.sqrt(np.sum(a[:dimen - j]*a[:dimen - j])*np.sum(b[:dimen - j]*b[:dimen - j]))
        corr.append(value/norm)
    return np.array(corr)




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
                    grad = False, filt = False, fps = 5.0, file_name = "All_result", change_frequency = False):
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
    return 

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


def load_one_density_data(prefix, sidelength = 38.0, width = 5, skip = 1, grad = False, start = 0, fps = 5.0, filt = False,
                          rearange = True, Q = 100):
    x0, y0, R = boundary(prefix)
    data = helpy.load_data(prefix)
    data['o'] = (data['o'] + np.pi)%(2 * np.pi)   # flip the detected orientation
    tracksets = helpy.load_tracksets(data, run_track_orient=True, min_length = 500, run_repair = 'interp')
    track_prefix = {prefix: tracksets}
    v_data = velocity.compile_noise(track_prefix, width=(width,), cat = False, side = sidelength, fps = fps, # fps or fps/skip need attention
                               ring = True, x0= x0, y0 = y0, skip = skip, grad = grad, start = start)
    v_data = v_data[prefix]
    if filt:
        for key, values in v_data.items():
            values['vring'] = filter_shaker_freq(values['vring'], fps = fps, Q = Q) 
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
    if skip != 1.0:
        for key, value in v_data.items():
            value['f'] /= 5
    fdata = helpy.load_framesets(v_data)
    np.save((prefix +'_v_data_'+str(width)+'.npy'), v_data)
    np.save((prefix +'_frame_data_'+str(width)+'.npy'), fdata)
    return fdata, v_data

#Individual Plot function
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
    heatmap = ax.pcolor(matrix, cmap=plt.get_cmap('gist_rainbow'), alpha=0.8)
    fig = plt.gcf()
    fig.set_size_inches(8,10)
    cbar = fig.colorbar(heatmap, shrink = 1.0,aspect= 40)
    cbar.ax.tick_params(labelsize=12)
    ax.set_frame_on(False)
    ax.grid(False)
    ax = plt.gca()
    ax.tick_params(axis = "both", which = "major", labelsize = 20)
    #ax.imshow(matrix, origin = "lower", cmap = "Blues", interpolation = "nearest")
    ax.xaxis.set_ticks(np.arange(0, 75, 15))
    ax.yaxis.set_ticks(np.arange(0,200, 40))
    ax.set_title("Velocity Colormap", fontsize = 24)
    ax.set_xlabel("Particle Index", fontsize = 24)
    ax.set_ylabel("Time(shake)", fontsize = 24)
    fig.savefig(prefix + name + "_colormap_" + str(width) + ".jpeg",dpi = 400)
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


def find_cor_length(corr, yToFind = 0):
    x = np.arange(len(corr))
    #f = interpolate.UnivariateSpline(x,corr, s=0)
    yreduced = np.array(corr) - yToFind
    freduced = interpolate.UnivariateSpline(x, yreduced, s=0)
    return freduced.roots()[0]


#Group Plot Function
###############################################################################

def diff_quantity_plot(file_name, save_name, quantity ,width = 5, vcompare = False, vcorr = False, dense = False, flow = True,normalize= True):
    """
    vcompare: v_theta, v_ring, v_omega comparison
    dense: if False plot v as a function of vacancy, if True v as a function of number
    flow: 1-D traffic flow rate
    normalize: normalize the correlation length by definition in coeff
    vcorr: different correlation between different v
    """
    direct_name = os.path.dirname(file_name)
    cor_den = np.load(file_name).item()
    fig, ax = plt.subplots()
    keys = cor_den.keys()
    number_keys = [eval(key.split('/')[-2]) for key in keys]
    max_number = np.max(number_keys)
    sort_index = np.argsort(number_keys)
    sort_keys = np.array(keys)[sort_index]
    NUM_COLORS = len(keys)
    cm = plt.get_cmap('gist_rainbow')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    for key in sort_keys:
        ax.plot(np.arange(20), cor_den[key]['time_correlation'][:20], 'o--',lw = 2.2 , label = key.split('/')[-2])
    ax.axhline(y = 0, color = 'black', linestyle = '--')
    ax.legend(fancybox=True, ncol = 2)
    ax.set_title("time correlation of different " + quantity, fontsize = 15)
    ax.set_xlabel(r"$\Delta t$(shake)", fontsize = 15)
    #ax.tick_params(axis='both', which='major', labelsize = 10)
    #ax.set_ylim([-0.2,0.4])
    fig.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_time_correlation.pdf")), dpi = 400)
    
    fig2, ax2 = plt.subplots()
    ax2.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    cor_length = list()
    id_number = list()
    flows = list()
    for key in sort_keys:
        correlation_result = cor_den[key]['correlation']
        numbers = int(key.split('/')[-2])
        if numbers != 6:
            id_number.append(numbers)
            coeff = float(max_number)/numbers if normalize else 1
            cor_length.append(find_cor_length(correlation_result) * coeff)
        flows.append(cor_den[key]['flow_rate'])
        ax2.plot(correlation_result, 'o--', lw = 2.5, label = key.split('/')[-2])
    ax2.axhline(y = 0, color = 'black', linestyle = '--')
    ax2.legend(fancybox=True, ncol = 2)
    #ax2.set_yscale('log')
    #ax2.set_xscale('log')
    ax2.set_title("Spatial correlation of different " + quantity, fontsize = 15)
    ax2.set_xlabel("Particle index j", fontsize = 15)
    #ax2.set_ylabel("correlation", fontsize = 15)
    #ax2.tick_params(axis='both', which='major', labelsize = 15)
    fig2.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_velocity_correlation.pdf")), dpi = 400)
    
    len_fig, len_ax = plt.subplots()
    len_ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    len_ax.set_xlabel("Vacancy", fontsize = 15)
    len_ax.set_ylabel("L", fontsize = 15)
    len_ax.set_title("Correlation length as a function of vacancy", fontsize = 15)
    #len_ax.tick_params(axis='both', which='major', labelsize = 15)
    vacancy = [(max_number - float(a))/max_number for a in id_number]
    for a, b in zip(id_number, cor_length):
        len_ax.scatter((max_number - float(a))/max_number, b,alpha = 0.6, label = a, s = 80)
    len_ax.legend(fancybox = True, ncol = 2)
    len_fig.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_correlation_length.pdf")), dpi = 400)
    np.save(os.path.join(direct_name, (str(normalize) +" correlation length.npy")),zip(vacancy, cor_length, id_number))

    ang_fig, ang_ax = plt.subplots(figsize = (10, 10))
    ang_ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    height = 0
    for key in sort_keys:
        ang_ax.plot(np.linspace(0,np.pi*2,sep_k)[:750], cor_den[key]['angular_distr'][:750] + height, alpha = 0.6, label = key.split('/')[-2])
        height += 1
    ang_ax.legend(fancybox = True, ncol =2)
    ang_ax.set_title("Angular Distribution Function" + str(sep_k))
    ang_ax.set_xlabel(r"$\theta$", fontsize = 15)
    ang_ax.set_ylabel(r"$g(\theta)$", fontsize = 15)
    ang_fig.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_angular_distribution.pdf")), dpi = 400)        


    peak_fig, peak_ax = plt.subplots(figsize = (10,10))
    peak_ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    height = 0
    for key in sort_keys:
        angular = cor_den[key]['angular_distr']
        angles_x = np.linspace(0,np.pi*2, sep_k)
        pt_number = float(key.split('/')[-2])
        indices, prop = find_peaks(angular, distance = np.floor(0.8*sep_k/pt_number))
        indices = np.insert(indices, 0, 0)
        half = int(np.floor(len(indices)/2.0))
        """
        x = angles_x[indices[:half]]
        y = np.repeat(height, len(x))
        peak_ax.scatter(np.arange(1, len(x)+1), x*pt_number/np.pi/2, label = pt_number)
        height+=1
        """
        mak = 'x' if pt_number == 60 else 'o'
        peak_ax.scatter(angles_x[indices[:half]], angular[indices[:half]] , label = pt_number, marker = mak)
#        - np.mean(angular[indices[:half]][-4:])
    peak_ax.legend(fancybox = True, ncol = 2)
    peak_ax.set_yscale('log')
#    peak_ax.set_xscale('log')
    peak_ax.set_ylim(bottom = 10**-3)
    peak_ax.set_xlim(left = 10**-1.5)
    peak_fig.savefig(os.path.join(direct_name, (save_name+ "_" + quantity + "_peak.pdf")), dpi = 400) 
    

    datas = defaultdict(dict)
    data_name = ['v_average', 'flow_rate']
    for name in data_name:
        for key in sort_keys:
            datas[name][float(key.split('/')[-2])] = cor_den[key][name]
    datas = dict(datas)
    np.save('/Users/zhejunshen/Result/APS/traffic/flow_data.npy', datas)
    
    if dense:
        fig3, ax3 = plt.subplots()
        ax3.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            ax3.scatter(key.split('/')[-2] ,np.abs(cor_den[key]['v_average']) , alpha = 0.6, label = key.split('/')[-2])
        ax3.set_title(r"Average $V_{\theta}$ of Different " + quantity)
        ax3.legend(fancybox=True, ncol = 2)
        fig3.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_average_velocity.pdf")), dpi = 400)
        
        fig4, ax4= plt.subplots()
        ax4.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            ax4.scatter(key.split('/')[-2] ,cor_den[key]['v_std'], alpha = 0.6, label = key.split('/')[-2])
        ax4.set_title(r"Std of $V_{\theta}$ of Different " + quantity)
        ax4.legend(fancybox=True, ncol = 2)
        fig4.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_velocity_std.pdf")),dpi = 400) 
    else:
        fig3, ax3 = plt.subplots()
        ax3.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            ax3.scatter((max_number - float(key.split('/')[-2]))/max_number ,
                        np.abs(cor_den[key]['v_average'])/0.0575 , 
                        alpha = 0.6, label = key.split('/')[-2], s= 80)
        print(np.abs(cor_den[key]['v_average']))
        ax3.set_ylabel(r"$V_{\theta}/(V_{\theta})_{single}$", fontsize = 16)
        ax3.set_title(r"$V_{\theta}/(V_{\theta})_{single}$ as a function of vacancy", fontsize = 16)
        ax3.set_xlabel("Vacancy", fontsize = 16)
        ax3.legend(fancybox=True, ncol = 2)
        fig3.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_average_velocity.pdf")), dpi = 400)
    
        fig4, ax4= plt.subplots()
        ax4.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            ax4.scatter((max_number - float(key.split('/')[-2])) /max_number,cor_den[key]['v_std'], alpha = 0.6, label = key.split('/')[-2])
        ax4.set_title(r"Std of $V_{\theta}$ as a function of vacancy")
        ax4.legend(fancybox=True, ncol = 2)
        fig4.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_velocity_std.pdf")),dpi = 400)     
     
        
    if flow:
        fig_flow, ax_flow = plt.subplots()
        ax_flow.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for a, b in zip(id_number, flows):
            ax_flow.scatter((max_number - float(a))/max_number, b, alpha = 0.6, s = 80, label = a)
        ax_flow.set_xlabel("Vacancy",fontsize = 16)
        ax_flow.set_ylabel("Flow Rate", fontsize = 16)
        ax_flow.set_title("Flow rate as a function of vacancy",fontsize = 16)
        ax_flow.legend(fancybox=True, ncol = 2)
        fig_flow.savefig(os.path.join(direct_name, (save_name+"_" + quantity + "_flow_rate.pdf")),dpi = 400)
            
    
    if vcompare:
        figure, axe = plt.subplots(2,figsize = (5,10))
        axe[0].set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        axe[1].set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            axe[0].scatter(cor_den[key]['vomega_average']*100, cor_den[key]['v_average'],  alpha = 0.6, label = key.split('/')[-2])
            axe[1].scatter(cor_den[key]['vradi_average']*100, cor_den[key]['v_average'], alpha = 0.6, label = key.split('/')[-2])
        axe[0].set_title(r"$V_{\theta}$ and $V_{\omega}$")
        axe[0].legend(ncol = 2)
        axe[1].set_title(r"$V_{\theta}$ and $V_{r}$")
        axe[1].legend(ncol = 2)
        figure.savefig(os.path.join(direct_name, (save_name+"_vrelation.pdf")),dpi = 400)
    
    
    if vcorr:
        fig5, ax5 = plt.subplots()
        fig6, ax6 = plt.subplots()
        ax5.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        ax6.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
        for key in sort_keys:
            vmatrix = cor_den[key]['v_matrix']
            vomega_matrix = cor_den[key]['vomega_matrix']
            vr_matrix = cor_den[key]['vr_matrix']
            for i in np.arange(vmatrix.shape[1]):
                if i == 0:
                    vcors = correlation_function(vmatrix[:,i], vomega_matrix[:,i])
                    vrcors = correlation_function(vmatrix[:,i], vr_matrix[:,i])
                else:
                    vcors = np.vstack((vcors, correlation_function(vmatrix[:,i], vomega_matrix[:,i])))
                    vrcors = np.vstack((vrcors, correlation_function(vmatrix[:,i], vr_matrix[:,i])))
            result = np.mean(vcors, axis = 0)
            v_vr_cor = np.mean(vrcors, axis = 0)
            
            ax5.plot(result[:20], 'o--', label = key.split('/')[-2])
            ax6.plot(v_vr_cor[:20], 'o--', label = key.split('/')[-2])
        ax5.axhline(y =0, color = 'black', ls ='--')    
        ax5.legend(fancybox = True, ncol = 2)
        ax6.axhline(y =0, color = 'black', ls ='--')    
        ax6.legend(fancybox = True, ncol = 2)
        
        
    
    return


# One density Analysis and Result Plot
###############################################################################
def vdata_analysis(prefix, width = 5, skip = 1, start = 0, grad = False, filt = False, fps = 5.0, Q = 100):
    """
    passes are the particles passes the 0 angle, note: noise would cause the particle moving back,
    therefore it's possible for one particle to pass 0 angle 2 times in a very short time interval.
    """
    # get vdata
    v_path = prefix + '_v_data_'+ str(width) + '.npy'
    if False:
    #if os.path.exists(v_path):
        vdata = np.load(v_path).item()
    else:
        vdata = load_one_density_data(prefix, width = width, skip = skip, 
                                      grad = grad, start = start, filt = filt, fps = fps, Q = Q)[1]
    #vstack vmatrix, axis = 0 is particle id,
    flow = 0
    for key, value in vdata.items():
        if key == 0:
            length = len(value['vring'])
            v_matrix = value['vring']
        else:
            if len(value['vring']) == length:
                v_matrix = np.vstack((v_matrix, value['vring']))
        codata = value['corient']
        passed = sum((codata[1:]-codata[:-1]) < -6)
        flow += passed
    #calculate time correlation 
    for i in np.arange(v_matrix.shape[0]):
        if i == 0:
            vcor = correlation_function(v_matrix[i], v_matrix[i])
        else:
            vcor = np.vstack((vcor, correlation_function(v_matrix[i], v_matrix[i])))
    
    vcor_average = np.mean(vcor, axis = 0)
    flow_rate = flow#/float(len(vdata[0]))*1000
    return vcor_average, v_matrix.T, flow_rate



def giant_fluct(fdata, x0, y0, R, sidelength = 28.0):
    ret = defaultdict(list)
    for f,framedata in fdata.items():
        position = framedata['xy'] - [x0,y0]
        rs = np.hypot(*position.T)
        for i in range(1, int(R / sidelength)):
            ret[i].append(sum(rs < i * sidelength))
    mean, flut = list(), list()
    for key, value in ret.items():
        mean.append(np.mean(value))
        flut.append(np.std(value))
    fig, ax = plt.subplots()
    ax.scatter(mean, flut)
    return 
    


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
    vr_time_op_t = np.empty(0)
    v_omega_t = np.empty(0)
    angle_dist = np.empty(0)
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
            v_matrix = layer_data['vring']  #vring_matrix
            #print(len(layer_data['vring']))
            v_err_matrix = layer_data['vring'] - layer_data['vring'].mean()
            v_corr = correlation_function(layer_data['vring'], layer_data['vring'])
            vr_matrix = layer_data['vradi']  #  radial velocity 
            vomega_matrix = layer_data['vomega']
            angle_dist = angular_distr(cen_orient, sep_k) # separation
            number = sum(mask)       
        else:
            v_matrix = np.vstack((v_matrix, layer_data['vring']))
            v_err_matrix = np.vstack((v_err_matrix, layer_data['vring'] - layer_data['vring'].mean()))
            v_corr = np.vstack((v_corr, correlation_function(layer_data['vring'], layer_data['vring'])))
            vr_matrix = np.vstack((vr_matrix, layer_data['vradi']))    # radial velocity 
            vomega_matrix = np.vstack((vomega_matrix, layer_data['vomega']))
            angle_dist = np.vstack((angle_dist, angular_distr(cen_orient, sep_k))) # separation
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
                'v_err_matrix', 'number', 'vomega_matrix', 'vr_matrix']
    ring_data_result = dict()
    for name in all_data:
        ring_data_result[name] = eval(name)
    time_correlation,flow_rate = vdata_analysis(prefix, width = width, skip = skip, 
                                      start = start, filt = filt, grad = grad, 
                                      fps = fps)[0::2]
    ring_data_result['time_correlation'] = time_correlation
    ring_data_result['v_average'] = np.nanmean(v_matrix)
    ring_data_result['v_std'] = np.nanstd(v_matrix)
    ring_data_result['vradi_std'] = np.nanstd(vr_matrix)
    ring_data_result['vomega_average'] = np.nanmean(vomega_matrix)
    ring_data_result['vomega_std'] = np.nanstd(vomega_matrix)
    ring_data_result['vradi_average'] = np.nanmean(vr_matrix)
    ring_data_result['flow_rate'] = flow_rate
    ring_data_result['angular_distr'] = np.nanmean(angle_dist, axis = 0)
    np.save(prefix + '_single_density_' + str(width) +'.npy',ring_data_result)
    return ring_data_result


        
        
    

def one_density_plots(prefix, width = 5, skip= 1, grad = False, filt = False, fps = 5.0):
    file_name = prefix + '_single_density_' + str(width) + '.npy' 
    if False:
    #if os.path.exists(file_name):
        ring_data_result = np.load(prefix + '_single_density_' + str(width) + '.npy').item()
    else:
        ring_data_result = one_density_analysis(prefix, width = width, skip = skip, grad = grad, filt = filt, fps = fps)
    print(ring_data_result['v_average'])
    vcor, vmatrix,flow_rate = vdata_analysis(prefix, width = width, skip = skip, grad = grad, filt = filt, fps = fps)
    cor2d = auto_correlation_function_2d(vmatrix)
    plot_2d_correlation(cor2d, prefix, width)
    plot_time_corr(vcor, prefix, ring_data_result['number'], width)
    plot_vcross(ring_data_result['correlation'], prefix, width)
    plot_std(ring_data_result['vstd_t'], prefix, width)
    plot_std_zoom(ring_data_result['vstd_t'], prefix, width)
    plot_velocity_map(ring_data_result['v_matrix'][:200], prefix, "_v", width)
    plot_velocity_map(ring_data_result['v_err_matrix'][:200], prefix, "_verror", width)
    plot_individual_velocity(ring_data_result['v_matrix'][:,1], prefix, 1, width)
    return 
    

def number_fluctuation(fdata, outer_range, size):
    import collections
    result = collections.defaultdict(list)
    for f, frame_data in fdata.items():
        if f % 5 == 0:
            for boundary in np.arange(outer_range, 0, -size):
                result[boundary].append(sum(frame_data['r'] < boundary))
    numbers = list()
    fluct = list()
    for key, value in result.items():
        numbers.append(np.mean(value))
        fluct.append(np.std(value))
    numbers = np.asarray(numbers)
    fluct = np.asarray(fluct)
    plt.scatter(numbers, fluct/np.sqrt(numbers))
    plt.ylabel(r'$\frac{\Delta{N}}{\sqrt{N}}$', size = 20)
    plt.xlabel(r'$N$',size = 20)
    return  
    



def layers_analysis(prefix, grad = True, fps =1.0, sidelength = 38.0):
    x0, y0, R = boundary(prefix)
    fdata = load_one_density_data(prefix, width = (0.575,), skip = 1, grad = grad, filt = False, fps = 1)[0]
    total_layer = int(np.floor(R/sidelength))
    layers_data = defaultdict(dict)
    for layer_number in np.arange(total_layer):
        vring_dist = np.empty(0) # vring_dist
        op_t = np.empty(0)  # op for each frame
        vr_time_op_dist= np.empty(0)  # vring * sign of op
        op_dist = np.empty(0)   # op distribution  in ring
        vring_t = np.empty(0)  # v for each frame in ring
        vstd_t = np.empty(0) # std of v for each frame in ring
        ps, ng, diff = list(), list(), list()
        """
        vp_t = np.empty(0) # v_p in bulk
        vp_dist = np.empty(0) #vp in bulk
        vt_dist = np.empty(0)  # vt in bulk
        v_par_matrix = np.empty(0) # vpar 
        """
        vr_time_op_t = np.empty(0)
        for f, framedata in fdata.iteritems():
            mask = ( framedata['r'] < R - layer_number * sidelength) & (framedata['r'] > R - (layer_number + 1) * sidelength)
            
            layer_data = framedata[mask]  # in layer
            cen_orient = layer_data['corient']
            cor_orient = layer_data['o']
            cen_unit_vector = np.array([np.cos(cen_orient), np.sin(cen_orient)]).T
            cor_unit_vector = np.array([np.cos(cor_orient), np.sin(cor_orient)]).T
            ring_orient = - np.cross(cen_unit_vector, cor_unit_vector)/np.sin(np.pi/4)
            
            op_dist = np.append(op_dist, ring_orient)  #op distribution
            vring_dist = np.append(vring_dist, layer_data['vring'])
            
            vstd_t = np.append(vstd_t, np.nanstd(layer_data['vring']))
            vring_t = np.append(vring_t, np.nanmean(layer_data['vring']))
            
            vr_time_op_dist = np.append(vr_time_op_dist, layer_data['vring'] * np.sign(ring_orient))
            vr_time_op_t = np.append(vr_time_op_t, np.nanmean(layer_data['vring'] * np.sign(ring_orient)))
            
            
            op_t = np.append(op_t, np.nanmean(ring_orient))
            
            ps.append(sum(ring_orient > 0))
            ng.append(sum(ring_orient < 0))
            diff.append(sum(ring_orient > 0) - sum(ring_orient < 0))
            
        all_data = ['vring_dist', 'op_t', 'vr_time_op_dist', 'op_dist', 'vring_t',
            'vstd_t', 'vr_time_op_t']
        for name in all_data:
            layers_data[layer_number + 1][name] = eval(name)
    np.save(prefix + '_layer_data_'  + '.npy', layers_data)
    return 
    

    