#!/usr/bin/env python
# encoding: utf-8

from __future__ import division
import numpy as np
from PIL import Image as Im
from itertools import izip
import sys

import helpy

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = ''#/Users/leewalsh/Physics/Squares/orientation/'
    extdir = locdir#'/Volumes/bhavari/Squares/lighting/still/'
    plot_capable = True
elif 'foppl' in hostname:
    computer = 'foppl'
    locdir = '/home/lawalsh/Granular/Squares/diffusion/orientational/'
    extdir = '/media/bhavari/Squares/diffusion/still/'
    import matplotlib
    matplotlib.use("agg")
    plot_capable = False
elif 'peregrine' in hostname:
    computer = 'peregrine'
    locdir = extdir = ''
    plot_capable = True
else:
    print "computer not defined"
    locdir = extdir = ''
    plot_capable = helpy.bool_input("Are you able to plot?")
    if plot_capable:
        import matplotlib

from matplotlib import pyplot as pl
from matplotlib import cm as cm

pi = np.pi
twopi = 2*pi

if __name__=='__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prefix', metavar='PRE',
                        help="Filename prefix with full or relative path "
                             "(filenames prefix_POSITIONS.txt, "
                             "prefix_CORNER_POSITIONS.txt, etc)")
    parser.add_argument('-c', '--corner', action='store_true',
                        help='Track corners instead of centers')
    parser.add_argument('-n', '--number', type=int, default=-1,
                        help='Total number of particles')
    parser.add_argument('-l','--load', action='store_true',
                        help='Create and save structured array from '
                             'prefix[_CORNER]_POSITIONS.txt file')
    parser.add_argument('-t','--track', action='store_true',
                        help='Connect the dots and save in the array')
    parser.add_argument('-p', '--plottracks', action='store_true',
                        help='Plot the tracks')
    parser.add_argument('-d', '--msd', action='store_true',
                        help='Calculate the MSD')
    parser.add_argument('--plotmsd', action='store_true',
                        help='Plot the MSD (requires --msd first)')
    parser.add_argument('-s', '--side', type=float, default=1,
                        help='Particle size in pixels, for unit normalization')
    parser.add_argument('-f', '--fps', type=int, default=1,
                        help="Number of frames per second (or per shake) "
                             "for unit normalization")
    parser.add_argument('--dt0', type=int, default=10,
                        help='Stepsize for time-averaging of a single '
                             'track at different time starting points')
    parser.add_argument('--dtau', type=int, default=1,
                        help='Stepsize for values of tau '
                             'at which to calculate MSD(tau)')
    parser.add_argument('--killflat', type=int, default=0,
                        help='Minimum growth factor for a single MSD track '
                             'for it to be included')
    parser.add_argument('--killjump', type=int, default=100000,
                        help='Maximum initial jump for a single MSD track '
                             'at smallest time step')
    parser.add_argument('--singletracks', type=int, nargs='*', default=xrange(1000),
                        help='identify single track ids to plot')
    parser.add_argument('--showtracks', action='store_true',
                        help='Show individual tracks')
    parser.add_argument('-v', '--verbose', action='count',
                        help='Print verbosity')

    args = parser.parse_args()

    prefix = args.prefix
    print 'using prefix', prefix
    dotfix = '_CORNER' if args.corner else ''

    gendata  =  args.load
    findtracks = args.track
    plottracks = args.plottracks
    findmsd = args.msd
    plotmsd = args.plotmsd

    S = args.side
    A = S**2
    fps = args.fps
    dtau = args.dtau
    dt0 = args.dt0

    kill_flats = args.killflat
    kill_jumps = args.killjump*S*S
    singletracks = args.singletracks
    show_tracks = args.showtracks
    verbose = args.verbose

else:
    verbose = False

def find_closest(thisdot, trackids, n=1, maxdist=20., giveup=10):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = max(trackids) + 1
        if verbose:
            print "New track:", newtrackid
            print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
        return newtrackid
    else:
        olddots = data[data['f']==frame-n]
        dists = ((thisdot['x'] - olddots['x'])**2 +
                 (thisdot['y'] - olddots['y'])**2)
        mini = np.argmin(dists)
        mindist = dists[mini]
        closest = olddots[mini]
        if mindist < maxdist:
            # a close one! Is there another dot in the current frame that's closer though?
            curdots = data[data['f']==frame]
            curdists = ((curdots['x'] - closest['x'])**2 +
                        (curdots['y'] - closest['y'])**2)
            mini2 = np.argmin(curdists)
            mindist2 = curdists[mini2]
            if mindist2 < mindist:
                # create new trackid to be deleted (or overwritten?)
                newtrackid = max(trackids) + 1
                if verbose:
                    print "found a closer child dot to the this dot's parent"
                    print "New track:", newtrackid
                    print '\tframe:', frame,'n:', n,
                    print 'dot:', thisdot['id'],
                    print 'closer:', curdots[mini2]['id']
                return newtrackid
            return trackids[closest['id']]
        elif n < giveup:
            return find_closest(thisdot, trackids, n=n+1,
                                maxdist=maxdist, giveup=giveup)
        else: # give up after giveup frames
            newtrackid = max(trackids) + 1
            if verbose:
                print "Recursed {} times, giving up.".format(n)
                print "New track:", newtrackid
                print '\tframe:', frame, 'n:', n, 'dot:', thisdot['id']
            return newtrackid

# Tracking
def gen_data(datapath):
    print "loading positions data from", datapath
    if  datapath.endswith('results.txt'):
        shapeinfo = False
        # imagej output (called *_results.txt)
        dtargs = {  'usecols' : [0,2,3,5],
                    'names'   : "id,x,y,f",
                    'dtype'   : [int,float,float,int]} \
            if not shapeinfo else \
                 {  'usecols' : [0,1,2,3,4,5,6],
                    'names'   : "id,area,mean,x,y,circ,f",
                    'dtype'   : [int,float,float,float,float,float,int]}
        data = np.genfromtxt(datapath, skip_header = 1,**dtargs)
        data['id'] -= 1 # data from imagej is 1-indexed
    elif datapath.endswith('POSITIONS.txt'):
        # positions.py output (called *_POSITIONS.txt)
        from numpy.lib.recfunctions import append_fields
        data = np.genfromtxt(datapath,
                             skip_header = 1,
                             names = "f,x,y,lab,ecc,area",
                             dtype = [int,float,float,int,float,int])
        ids = np.arange(len(data))
        data = append_fields(data, 'id', ids, usemask=False)
    else:
        print "is {} from imagej or positions.py?".format(datapath.split('/')[-1])
        print "Please rename it to end with _results.txt or _POSITIONS.txt"
    return data

def find_tracks(data, n=-1, giveup=10):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 2*giveup))

    trackids = -np.ones(data.shape, dtype=int)
    if n==-1:
        n = len(data['f']==0)
        print "number of particles:", n

    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i], trackids, giveup=giveup)

    # save the data record array and the trackids array
    print "saving track data"
    # Michael used the data['lab'] field (as line[3] for line in data) to store
    # trackids. I'll keep doing that:
    assert len(data) == len(trackids), "too few/many trackids"
    assert np.allclose(data['id'], np.arange(len(data))), "gap in particle id"
    data['lab'] = trackids
    #data = data[trackids < n] # michael did this to "crop out extra tracks"


    np.savez(locdir+prefix+dotfix+"_TRACKS",
            data=data, trackids=trackids)

    return trackids

# Plotting tracks:
def plot_tracks(data, trackids, bgimage=None, mask=slice(None), fignum=None):
    pl.figure(fignum)
    data = data[mask]
    trackids = trackids[mask]
    pl.scatter(data['y'], data['x'],
            c=np.array(trackids)%12, marker='o', alpha=.5, lw=0)
    if bgimage:
        pl.imshow(bgimage, cmap=cm.gray, origin='upper')
    pl.gca().set_aspect('equal')
    pl.xlim(data['y'].min()-10, data['y'].max()+10)
    pl.ylim(data['x'].min()-10, data['x'].max()+10)
    pl.title(prefix)
    print "saving tracks image to", prefix+"_tracks.png"
    pl.savefig(locdir+prefix+"_tracks.png")
    pl.show()

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

def trackmsd(track, dt0, dtau):
    """ trackmsd(track, dt0, dtau)
        finds the track msd, as function of tau,
        averaged over t0, for one track (worldline)
    """
    trackdots = data[trackids==track]

    if dt0 == dtau == 1:
        if verbose: print "Using correlation"
        import correlation.msd
        xy = np.column_stack([trackdots['x'], trackdots['y']])
        return correlation.msd(xy, ret_taus=True)

    trackbegin, trackend = trackdots['f'][[0,-1]]
    tracklen = trackend - trackbegin + 1
    if verbose:
        print "tracklen =",tracklen
        print "\t from %d to %d"%(trackbegin, trackend)
    if isinstance(dtau, float):
        taus = helpy.farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        taus = xrange(dtau, tracklen, dtau)

    tmsd = []
    for tau in taus:  # for tau in T, by factor dtau
        #print "tau =", tau
        avg = t0avg(trackdots, tracklen, tau)
        #print "avg =", avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]])
    if verbose:
        print "\t...actually", len(tmsd)
    return tmsd

def t0avg(trackdots, tracklen, tau):
    """ t0avg() averages over all t0, for given track, given tau """
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        #print "t0=%d, tau=%d, t0+tau=%d, tracklen=%d"%(t0,tau,t0+tau,tracklen)
        olddot = trackdots[trackdots['f']==t0]
        newdot = trackdots[trackdots['f']==t0+tau]
        if len(newdot) != 1 or len(olddot) != 1:
            continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            if verbose > 1: print 'unflattened'
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
            if verbose: print 'flattened once'
            totsqdisp += sqdisp[0]
        else:
            if verbose: print "fail"
            continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

def find_msds(dt0, dtau, tracks=None):
    """ Calculates the MSDs"""
    print "Begin calculating MSDs"
    msds = []
    msdids = []
    if tracks is None:
        tracks = np.unique(trackids)
    for trackid in tracks:
        if verbose: print "calculating msd for track", trackid
        tmsd = trackmsd(trackid, dt0, dtau)
        if len(tmsd) > 1:
            tmsdarr = np.asarray(tmsd)
            msds.append(tmsd)
            msdids.append(trackid)
    return msds, msdids

# Mean Squared Displacement:

def mean_msd(msds, taus, msdids=None, kill_flats=0, kill_jumps=1e9,
             show_tracks=False, singletracks=xrange(1000), tnormalize=False,
             errorbars=False, fps=1, A=1):
    """ return the mean of several track msds """

    msd = np.full((len(msds),len(taus)), np.nan, float)
    added = np.zeros(len(taus), float)

    if msdids is not None:
        allmsds = izip(xrange(len(msds)), msds, msdids)
    elif msdids is None:
        allmsds = enumerate(msds)
    for thismsd in allmsds:
        if msdids is not None:
            ti, tmsd, msdid = thismsd
        else:
            ti, tmsd = thismsd
        if len(tmsd) < 2: continue
        tmsdt, tmsdd = np.asarray(tmsd).T
        if tmsdd[-50:].mean() < kill_flats: continue
        if tmsdd[:2].mean() > kill_jumps: continue
        if show_tracks:
            if msdids is not None and msdid not in singletracks: continue
            if tnormalize:
                pl.loglog(tmsdt/fps, tmsdd/A/(tmsdt/fps)**tnormalize)
            else:
                pl.loglog(tmsdt/fps, tmsdd/A, lw=0.5, alpha=0.5,
                          label=msdid if msdids is not None else '')
        tau_match = np.searchsorted(taus, tmsdt)
        msd[ti, tau_match] = tmsdd
    if errorbars:
        added = np.sum(np.isfinite(msd), 0)
        msd_err = np.nanstd(msd, 0) / np.sqrt(added)
    if show_tracks:
        pl.plot(taus/fps, (msd/(taus/fps)**tnormalize).T/A)
    msd = np.nanmean(msd, 0)
    return (msd, msd_err) if errorbars else msd

def plot_msd(msds, msdids, dtau, dt0, nframes, tnormalize=False, prefix='',
        show_tracks=True, figsize=(5,3), plfunc=pl.semilogx, meancol='',
        title=None, xlim=None, ylim=None, fignum=None, errorbars=False,
        lw=1, singletracks=xrange(1000), fps=1, S=1, ang=False, sys_size=0,
        kill_flats=0, kill_jumps=1e9, show_legend=False, save=''):
    """ Plots the MS(A)Ds """
    print "using dtau = {}, dt0 = {}".format(dtau, dt0)
    A = 1 if ang else S**2
    print "using S = {} pixels, thus A = {} px^2".format(S, A)
    try:
        dtau = np.asscalar(dtau)
    except AttributeError:
        pass
    if isinstance(dtau, (float, np.float)):
        taus = helpy.farange(dt0, nframes-1, dtau)
    elif isinstance(dtau, (int, np.int)):
        taus = np.arange(dtau, nframes-1, dtau)
    pl.figure(fignum, figsize)

    # Get the mean of msds
    msd = mean_msd(msds, taus, msdids,
            kill_flats=kill_flats, kill_jumps=kill_jumps, show_tracks=show_tracks,
            singletracks=singletracks, tnormalize=tnormalize, errorbars=errorbars,
            fps=fps, A=A)
    if errorbars: msd, msd_err = msd
    #print "Coefficient of diffusion ~", msd[np.searchsorted(taus, fps)]/A
    #print "Diffusion timescale ~", taus[np.searchsorted(msd, A)]/fps

    if tnormalize:
        plfunc(taus/fps, msd/A/(taus/fps)**tnormalize, 'ko',
               label="Mean Sq {}Disp/Time{}".format(
                     "Angular " if ang else "",
                     "^{}".format(tnormalize) if tnormalize != 1 else ''))
        plfunc(taus/fps, msd[0]/A*(taus/fps)**(1-tnormalize)/dtau,
               'k-', label="ref slope = 1", lw=2)
        plfunc(taus/fps, (twopi**2 if ang else 1)/(taus/fps)**tnormalize,
               'k--', lw=2, label=r"$(2\pi)^2$" if ang else
               ("One particle area" if S>1 else "One Pixel"))
        pl.ylim([0, 1.3*np.max(msd/A/(taus/fps)**tnormalize)])
    else:
        pl.loglog(taus/fps, msd/A, meancol, lw=lw,
                  label=prefix+'\ndt0=%d dtau=%d'%(dt0,dtau))
        pl.loglog(taus/fps, msd[0]/A*taus/dtau/2, meancol+'--', lw=2,
                  label="slope = 1")
    if errorbars:
        pl.errorbar(taus/fps, msd/A/(taus/fps)**tnormalize,
                    msd_err/A/(taus/fps)**tnormalize,
                    fmt=meancol, errorevery=errorbars)
    if sys_size:
        pl.axhline(sys_size, ls='--', lw=.5, c='k', label='System Size')
    pl.title("Mean Sq {}Disp".format("Angular " if ang else "") if title is None else title)
    pl.xlabel('Time (' + ('s)' if fps > 1 else 'frames)'), fontsize='x-large')
    if ang:
        pl.ylabel('Squared Angular Displacement ($rad^2$)',
              fontsize='x-large')
    else:
        pl.ylabel('Squared Displacement ('+('particle area)' if S>1 else 'square pixels)'),
              fontsize='x-large')
    if xlim is not None:
        pl.xlim(*xlim)
    if ylim is not None:
        pl.ylim(*ylim)
    if show_legend: pl.legend(loc='best')
    if save is None:
        save = locdir + prefix + "_MS{}D.pdf".format('A' if ang else '')
    if save:
        print "saving to", save
        pl.savefig(save)
    pl.show()

if __name__=='__main__':
    if findmsd:
        msds, msdids = find_msds(dt0, dtau)
        np.savez(locdir+prefix+"_MSD",
                 msds = np.asarray(msds),
                 msdids = np.asarray(msdids),
                 dt0  = np.asarray(dt0),
                 dtau = np.asarray(dtau))
        print "saved msd data to", prefix+"_MSD.npz"
    elif plotmsd:
        print "loading msd data from npz files"
        msdnpz = np.load(locdir+prefix+"_MSD.npz")
        msds = msdnpz['msds']
        try: msdids = msdnpz['msdids']
        except KeyError: msdids = None
        try:
            dt0  = np.asscalar(msdnpz['dt0'])
            dtau = np.asscalar(msdnpz['dtau'])
        except KeyError:
            dt0  = 10 # here's assuming...
            dtau = 10 #  should be true for all from before dt* was saved
        print "\t...loaded"

    if gendata:
        datapath = locdir+prefix+dotfix+'_POSITIONS.txt'
        data = gen_data(datapath)
        print "\t...loaded"
    if findtracks:
        if not gendata:
            data = np.load(locdir+prefix+'_POSITIONS.npz')['data']
        trackids = find_tracks(data, n=args.number)
    elif gendata:
        print "saving data only (no tracks) to "+prefix+dotfix+"_POSITIONS.npz"
        np.savez(locdir+prefix+dotfix+"_POSITIONS",
                data = data)
        print '\t...saved'
    else:
        # assume existing tracks.npz
        try:
            tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
            trackids = tracksnpz['trackids']
            print "loading data and tracks from "+prefix+"_TRACKS.npz"
        except IOError:
            tracksnpz = np.load(locdir+prefix+"_POSITIONS.npz")
            print "loading positions data from "+prefix+"_POSITIONS.npz"
        data = tracksnpz['data']
        print "\t...loaded"

if __name__=='__main__' and plot_capable:
    if plotmsd:
        print 'plotting now!'
        plot_msd(msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
                 prefix=prefix, show_tracks=show_tracks,
                 singletracks=singletracks, fps=fps, S=S,
                 kill_flats=kill_flats, kill_jumps=kill_jumps)
    if plottracks:
        try:
            bgimage = Im.open(extdir+prefix+'_0001.tif')
        except IOError:
            try:
                bgimage = Im.open(locdir+prefix+'_001.tif')
            except IOError:
                bgimage = None
        if singletracks:
            mask = np.in1d(trackids, singletracks)
        plot_tracks(data, trackids, bgimage, mask=mask)
