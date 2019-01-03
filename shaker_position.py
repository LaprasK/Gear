#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 17:31:03 2017

@author: zhejun
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import regionprops
from collections import namedtuple, defaultdict
from positions import label_particles_walker
import os
from skimage.measure import label
from skimage.morphology import label as sklabel
from scipy.optimize import curve_fit

Segment = namedtuple('Segment', 'x y label area'.split())


def prep_image(im, width=2):
    im = im[400:800]  # pick desired region here
    im = im.T
    s = width*im.std()
    m = im.mean()
    im = im -(m - s)
    im /= 2*s
    np.clip(im, 0, 1, out=im)
    return im

def filter_segments(labels, min_area = 300, max_area = 1500):
    pts = list()
    rpropargs = labels, None
    for rprop in regionprops(*rpropargs):
        area = rprop['area']
        good = min_area <= area <= max_area
        if not good:
            continue
        x, y = rprop["Centroid"]
        pts.append(Segment(x, y, rprop.label, area))
    return pts

from skimage.morphology import disk as skdisk
from scipy import ndimage

def disk(n):
    """create a binary array with a disk of size `n`"""
    return skdisk(n).astype(int)


def gdisk(width, inner=0, outer=None):
    """ create a gaussian kernel with constant central disk, zero sum, std dev 1

        shape is a disk of constant value and radius `inner`, which falls off as
        a gaussian with `width`, and is truncated at radius `outer`.

        parameters
        ----------
        width : width (standard dev) of gaussian (approx half-width at half-max)
        inner : radius of constant disk, before gaussian falloff (default 0)
        outer : full radius of nonzero part, beyond which array is truncated
            (default outer = inner + 2*width)

        returns
        -------
        gdisk:  a square array with values given by
                    / max for r <= inner
            g(r) = {  min + (max-min)*exp(.5*(r-inner)**2 / width**2)
                    \ 0 for r > outer
            min and max are set so that the sum of the array is 0 and std is 1
    """
    outer = outer or inner + 4*width
    circ = disk(outer)
    incirc = circ.nonzero()

    x = np.arange(-outer, outer+1, dtype=float)
    x, y = np.meshgrid(x, x)
    r = np.hypot(x, y) - inner
    np.clip(r, 0, None, r)

    g = np.exp(-0.5*(r/width)**2)
    g -= g[incirc].mean()
    g /= g[incirc].std()
    g *= circ
    return g


def label_particles_convolve(im, kern, thresh=3, rmv=None, **extra_args):
    """ Segment image using convolution with gaussian kernel and threshold

        parameters
        ----------
        im : the original image to be labeled
        kern : kernel size
        thresh : the threshold above which pixels are included
            if thresh >= 1, in units of intensity std dev
            if thresh < 1, in absolute units of intensity
        rmv : if given, the positions at which to remove large dots

        returns
        -------
        labels : an image array of uniquely labeled segments
        convolved : the convolved image before thresholding and segementation
    """
    kernel = np.sign(kern)*gdisk(abs(kern)/4, abs(kern))
    convolved = ndimage.convolve(im, kernel)
    convolved -= convolved.min()
    convolved /= convolved.max()

    if thresh >= 1:
        if rmv is not None:
            thresh -= 1  # smaller threshold for corners
        thresh = thresh*convolved.std() + convolved.mean()
    threshed = convolved > thresh

    labels = sklabel(threshed, connectivity=1)
    return labels, convolved


def shake(prefix, cutoff = 200, method = "convolve", manual = False):
    tifs = list()
   # tifs= map('image_{:04d}.tif'.format,  range(cutoff))
    allfiles = os.listdir(prefix)[:cutoff]
    for tif in allfiles:
        if tif.endswith(".tif"):
            tifs.append(os.path.join(prefix, tif))
    labels = defaultdict(list)
    for tif in tifs:
        im = plt.imread(tif)
        im = prep_image(im)
        if method == "convolve":
            filt = label_particles_convolve(im, -20)[0]
            label_image = filt
        elif method == "segment":
            filt = label_particles_walker(im)
            label_image = label(filt)
        pts = filter_segments(label_image)
        pts = np.array(pts, dtype = object)
        if not manual:            
            for pt in pts:
                labels[pt[2]].append(pt[1])
        if manual:
            for pt in pts:
                if np.abs(pt[0] - 85) < 5:
                    labels[0].append(pt[1])
                elif np.abs(pt[0] - 312) < 5:
                    labels[1].append(pt[1])
                elif np.abs(pt[0] - 387) < 5:
                    labels[2].append(pt[1])
                elif np.abs(pt[0] - 868) < 5:
                    labels[3].append(pt[1])
                elif np.abs(pt[0] - 929) < 5:
                    labels[4].append(pt[1]) 
                elif np.abs(pt[0] - 1163) < 5:
                    labels[5].append(pt[1])
        filt_center = label_particles_convolve(im, 20)[0]
        pts = filter_segments(filt_center, min_area=1300, max_area= 10000)
        pts = np.array(pts, dtype = object)        
        labels[6].append(pts[0][1])
    return labels



        
def plot_result(result, name):
    colors = ["r","g","b","yellow","black","cyan", "pink"]
    i = 1
    for key, value in result.items():
        v = np.asarray(value)
        result[key] = v - v.mean() + i
        i += 1
    fig, axes = plt.subplots()
    i = 0
    for key, value in result.items():
        axes.plot(value, colors[i])
        i += 1
    fig.savefig("/Users/zhejun/Document/Result/" + name + ".pdf")
    return


def sinfunc(t, A, w, p, c):  
    return A * np.sin(w*t + p) + c

def fit_sin(yy):
    '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
    tt = np.arange(len(yy))
    yy = np.array(yy)
    ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])
    popt, pcov = curve_fit(sinfunc, tt, yy, p0=guess)
    return popt

def check_freq(labels):
    freq = dict()
    phase = dict()
    ampl = dict()
    for key, value in labels.items():
        a, omega, p = fit_sin(value)[:3]
        phase[key] = p
        ampl[key] = a
        freq[key] = 2*np.pi/omega
    return ampl, freq, phase
    
        
        
        