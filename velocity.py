#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
from collections import defaultdict
from glob import iglob
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

import helpy
import tracks
import correlation as corr

description = """This script plots a histogram of the velocity noise for one or
several data sets. Includes option to subtract v_0 from translational noise.
The histogram figure is optionally saved to file prefix.plothist[orient].pdf
Run from the folder containing the positions file.
Copyright (c) 2015 Sarah Schlossberg, Lee Walsh; all rights reserved.
"""

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=description)
    arg = parser.add_argument
    arg('prefix', help='Prefix without trial number')
    arg('-o', '--orientation', action='store_false',
        dest='do_translation', help='Only orientational noise?')
    arg('-t', '--translation', action='store_false',
        dest='do_orientation', help='Only translational noise?')
    arg('--sets', type=int, default=0, metavar='N', nargs='?', const=1,
        help='Number of sets')
    arg('--width', type=float, default=(0.75,), metavar='W', nargs='*',
        help='Smoothing width for derivative, may give several')
    arg('--particle', type=str, default='', help='Particle name')
    arg('--save', type=str, nargs='?', const='velocity', default='',
        help='Save figure (optionally provide suffix)?')
    arg('--lin', action='store_false', dest='log', help='Plot on linear scale?')
    arg('--log', action='store_true', help='Plot on a log scale?')
    arg('--dupes', action='store_true', help='Remove duplicates from tracks')
    arg('--normalize', action='store_true', help='Normalize by max?')
    arg('--autocorr', action='store_true', help='Plot <vv> autocorrelation?')
    arg('--untrackorient', action='store_false', dest='torient',
        help='Untracked raw orientation (mod 2pi)?')
    arg('--interp', action='store_true', dest='interp', help='Interpolate gaps')
    arg('--stub', type=int, default=10, help='Min track length. Default: 10')
    arg('--nosubtract', action='store_false', dest='subtract',
        help="Don't subtract v0?")
    arg('-s', '--side', type=float,
        help='Particle size in pixels, for unit normalization')
    arg('-f', '--fps', type=float, help="Number frames per shake "
        "(or second) for unit normalization.")
    arg('-v', '--verbose', action='count', help="Be verbose")
    args = parser.parse_args()

pi = np.pi


def noise_derivatives(tdata, width=(1,), side=1, fps=1, xy=False,
                      do_orientation=True, do_translation=True, subtract=True):
    x = tdata['f']/fps
    ret = {}
    if do_orientation:
        ret['o'] = np.array([helpy.der(tdata['o'], x=x, iwidth=w)
                             for w in width]).squeeze()
    if do_translation:
        cos, sin = np.cos(tdata['o']), np.sin(tdata['o'])
        vx, vy = [np.array([helpy.der(tdata[i]/side, x=x, iwidth=w)
                            for w in width]).squeeze() for i in 'xy']
        if xy:
            ret['x'], ret['y'] = vx, vy
        else:
            vI = vx*cos + vy*sin
            vT = vx*sin - vy*cos
            ret['par'], ret['perp'] = vI, vT
        if subtract:
            v0 = vI.mean(-1, keepdims=vI.ndim > 1)
            if xy:
                ret['etax'] = vx - v0*cos
                ret['etay'] = vy - v0*sin
            else:
                ret['etapar'] = vI - v0
    return ret


def compile_noise(prefixes, vs, width=(1,), side=1, fps=1, cat=True,
                  do_orientation=True, do_translation=True, subtract=True,
                  stub=10, torient=True, interp=True, dupes=False, **ignored):
    print sorted(ignored)
    if np.isscalar(prefixes):
        prefixes = [prefixes]
    for prefix in prefixes:
        if args.verbose:
            print "Loading data for", prefix
        data = helpy.load_data(prefix, 'tracks')
        if dupes:
            data['t'] = tracks.remove_duplicates(data['t'], data)
        tracksets = helpy.load_tracksets(data, min_length=stub,
                                         run_track_orient=torient,
                                         run_fill_gaps=interp)
        for track in tracksets:
            tdata = tracksets[track]
            velocities = noise_derivatives(
                tdata, width=width, side=side, fps=fps, subtract=subtract,
                do_orientation=do_orientation, do_translation=do_translation)
            for v in velocities:
                vs[v].append(velocities[v])
    if cat:
        for v in vs:
            vs[v] = np.concatenate(vs[v], -1)
    return len(tracksets)


def get_stats(a):
    """Computes mean, D_T or D_R, and standard error for a list.
    """
    a = np.asarray(a)
    n = a.shape[-1]
    M = np.nanmean(a, -1, keepdims=a.ndim > 1)
    # c = a - M
    # variance = np.einsum('...j,...j->...', c, c)/n
    variance = np.nanvar(a, -1, keepdims=a.ndim > 1)
    D = 0.5*variance
    SE = np.sqrt(variance)/sqrt(n - 1)
    return M, D, SE


def compile_widths(prefixes, **compile_args):
    stats = {v: {s: np.empty_like(compile_args['width'])
                 for s in 'mean var stderr'.split()}
             for v in 'o par perp etapar'.split()}
    vs = defaultdict(list)
    compile_noise(prefixes, vs, **compile_args)
    for v, s in stats.items():
        s['mean'], s['var'], s['stderr'] = get_stats(vs[v])
    return stats


def plot_widths(widths, stats, normalize=False):
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$',
             'etapar': r'$\eta_\parallel$'}
    fig = plt.figure(figsize=(8, 12))
    for i, s in enumerate(stats['o']):
        ax = fig.add_subplot(len(stats['o']), 1, i+1)
        for v in stats:
            val = stats[v][s]
            if normalize:
                sign = np.sign(val.sum())
                val = sign*val
                val = val/val.max()
                ax.axhline(1, lw=0.5, c='k', ls=':', alpha=0.5)
            ax.plot(widths, val, '.'+ls[v]+cs[s], label=label[v])
        ax.set_title(s)
        ax.margins(y=0.1)
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
        if normalize:
            ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='best')
    return fig


def plot_hist(a, nax=1, axi=1, bins=100, log=True, orient=False, label='v',
              title='', subtitle=''):
    stats = get_stats(a)
    ax = axi[0] if isinstance(axi, tuple) else plt.subplot(nax, 2, axi*2-1)
    bins = ax.hist(a, bins, log=False, alpha=0.7,
            label=('$\\langle {} \\rangle = {:.5f}$\n'
                   '$D = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(label, *stats))[1]
    ax.legend(loc='upper left', fontsize='xx-small', frameon=False)
    ax.set_ylabel('Frequency')
    if orient:
        l, r = ax.set_xlim(bins[0], bins[-1])
        xticks = np.linspace(l, r, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(map('${:.2f}\pi$'.format, xticks/pi), fontsize='small')
    xlabel = 'Velocity ({}/vibration)'.format('rad' if orient else 'particle')
    ax.set_xlabel(xlabel)
    ax.set_title("{} ({})".format(title, subtitle), fontsize='medium')

    ax2 = axi[1] if isinstance(axi, tuple) else plt.subplot(nax, 2, axi*2)
    bins = ax2.hist(a, bins*2, log=True, alpha=0.7)[1]
    if orient:
        l, r = ax2.set_xlim(bins[0], bins[-1])
        xticks = np.linspace(l, r, 9)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(map('${:.2f}\pi$'.format, xticks/pi), fontsize='small')
    return ax, ax2

def vv_autocorr(prefixes, corrlen=0.5, **compile_args):
    vs = defaultdict(list)
    compile_noise(prefixes, vs, cat=False, **compile_args)
    vvs = {}
    for v, tvs in vs.iteritems():
        vcorrlen = int(corrlen*max(map(len, tvs))) if corrlen < 1 else corrlen
        vv = np.full((len(tvs), vcorrlen), np.nan, float)
        for i, tv in enumerate(tvs):
            ac = corr.autocorr(tv, norm=1, cumulant=False)
            vv[i, :len(ac)] = ac[:corrlen]
        vvcount = np.isfinite(vv).sum(0)
        vv = vv[:, vvcount > 0]
        vv = np.nanmean(vv, 0)
        dvv = np.nanstd(vv, 0)/np.sqrt(vvcount)
        vvs[v] = vv, dvv
    return vvs


if __name__ == '__main__':
    helpy.save_log_entry(args.prefix, 'argv')
    suf = '_TRACKS.npz'
    if '*' in args.prefix or '?' in args.prefix:
        fs = iglob(args.prefix+suf)
    else:
        dirname, prefix = os.path.split(args.prefix)
        dirm = (dirname or '*') + (prefix + '*/')
        basm = prefix.strip('/._')
        fs = iglob(dirm + basm + '*' + suf)
    prefixes = [s[:-len(suf)] for s in fs] or args.prefix

    helpy.save_log_entry(args.prefix, 'argv')
    meta = helpy.load_meta(args.prefix)
    if args.verbose:
        print 'using'
        print '\n'.join([prefixes] if np.isscalar(prefixes) else prefixes)
    helpy.sync_args_meta(args, meta, ['side', 'fps'],
                         ['sidelength', 'fps'], [1, 1])
    compile_args = dict(args.__dict__)

    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$',
             'etapar': r'$\eta_\parallel$'}
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    if len(args.width) > 1:
        stats = compile_widths(prefixes, **compile_args)
        plot_widths(args.width, stats, normalize=args.normalize)
    elif args.autocorr:
        vvs = vv_autocorr(prefixes, corrlen=10*args.fps, **compile_args)
        fig, ax = plt.figure()
        for v in vvs:
            vv, dvv = vvs[v]
            t = np.arange(len(vv))/args.fps
            ax.errorbar(t, vv, yerr=dvv, label=label[v], ls=ls[v])
        ax.set_title(r"Velocity Autocorrelation $\langle v(t) v(0) \rangle$")
        ax.legend(loc='best')
    else:
        vs = defaultdict(list)
        trackcount = compile_noise(prefixes, vs, **compile_args)

        nax = args.do_orientation + args.do_translation*(args.subtract + 1)
        axi = 1
        subtitle = args.particle
        bins = np.linspace(-1, 1, 51)
        if args.do_orientation:
            plot_hist(vs['o'], nax, axi, bins=bins*pi/2, log=args.log,
                      label=r'\xi', orient=True, title='Orientation',
                      subtitle=subtitle)
            axi += 1
        if args.do_translation:
            ax, ax2 = plot_hist(vs['par'], nax, axi, log=args.log,
                                label='v_\parallel', bins=bins)
            plot_hist(vs['perp'], nax, (ax, ax2), log=args.log, label='v_\perp',
                      bins=bins, title='Parallel & Transverse',
                      subtitle=subtitle)
            axi += 1
            if args.subtract:
                plot_hist(np.concatenate([vs['etapar'], vs['perp']]), nax, axi,
                          log=args.log, label=r'\eta_\alpha', bins=bins,
                          title='$v_0$ subtracted', subtitle=subtitle)
                axi += 1

    if args.save:
        savename = os.path.abspath(args.prefix.rstrip('/._?*'))
        savename += '_' + args.save + '.pdf'
        print 'Saving plot to {}'.format(savename)
        plt.savefig(savename)
    else:
        plt.show()
