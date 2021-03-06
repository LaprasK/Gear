#!/usr/bin/env python
import sys
import numpy as np
from scipy.spatial import KDTree, Voronoi
from math import sqrt
import math

def get_angle(p, q):
    # angle of (q - p) from the horizontal
    r = (q[0] - p[0], q[1] - p[1])
    return math.atan2(r[1], r[0])

def exp(C):
    # C complex
    return math.exp(C.real) * (math.cos(C.imag) + 1j * math.sin(C.imag))

def poly_area(corners):
    # calculate area of polygon
    area = 0.0
    n = len(corners)
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0

def find_COM(frame):
    return (sum([p[0] for p in frame]) / len(frame),
            sum([p[1] for p in frame]) / len(frame))

def take_avg(stat, ignore_first):
    if ignore_first:
        for frame in stat:
            frame[2].extend(frame[1])
            frame[1] = []
    ret = [[sum(l)/len(l) if len(l)>0 else -1.
            for l in frame[1:]] for frame in stat]
    return [x[1:] for x in ret] if ignore_first else ret

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please specify a filename.")
        sys.exit(0)

    M = 4 # number of neighbors

    fname = sys.argv[1]
    data = np.genfromtxt(fname + "_POSITIONS.txt", dtype="i,f,f,i,f,i",
                         names="f,x,y,lab,ecc,area")
    frames = [[(row[1], row[2]) for row in data[data['f']==i]]
              for i in range(data[-1][0] + 1)]
    frame_IDs = [[row[3] for row in data[data['f']==i]]
                 for i in range(data[-1][0] + 1)]
    psi_data = []
    frame_densities = []
    radial_psi = []
    radial_densities = []
    radial_r = []
    valencies = {}
    initial_pos = {}

    # Calculate valency for each ID based on first frame
    COM = find_COM(frames[0])
    dists = [((row[1]-COM[0])**2 + (row[2]-COM[1])**2, row[3])
             for row in data[data['f']==0]]
    sorted_dists = sorted(dists)
    n = 0
    i = 0
    while True:
        n += 1
        box_size = 8 * (n - 1) if n > 1 else 1
        for j in range(box_size):
            valencies[sorted_dists[i][1]] = n
            i += 1
            if i >= len(frames[0]):
                break
        else:
            continue
        break

    max_valency = max(valencies.values())
    # Find initial positions for each ID
    import pdb; pdb.set_trace()

    for j, frame in enumerate(frames):
        vor = Voronoi(frame)
        areas = []
        r_densities = []
        r_psi = []
        r_r = []
        r_msd = []
        for v in range(max_valency + 1): # each list contains all s=k
            r_densities.append([])
            r_psi.append([])
            r_r.append([])
            r_msd.append([])

        for i, p in enumerate(vor.points):
            region = vor.regions[vor.point_region[i]]
            if -1 in region: # infinite Voronoi cell
                continue
            areas.append(poly_area([vor.vertices[q] for q in region]))
            if areas[-1] > 0.:
                r_densities[valencies[frame_IDs[j][i]]].append(1. / areas[-1])

        areas = np.asarray(areas)
        densities = 1. / areas[areas > 0.]
        frame_densities.append(densities)
        radial_densities.append(r_densities)

        tree = KDTree(frame)
        COM = find_COM(frame)
        psi_frame = []
        for i, p in enumerate(frame):
            query_ret = tree.query([p], k=M+1)

            # remove p from the neighbors list by slicing
            neighbors = [tree.data[x] for x in query_ret[1][0]][1:]
            # if p is an edge or corner, remove extra neighbors
            min_dist = min((n[0]-p[0])**2 + (n[1]-p[1])**2 for n in neighbors)
            thresh = min_dist * 2 * .9 # slightly less than a diagonal
            neighbors = [n for n in neighbors
                         if (n[0]-p[0])**2 + (n[1]-p[1])**2 < thresh]
            N = len(neighbors)
            psi = sum(exp(M * get_angle(p, n) * 1j) for n in neighbors) / N
            valency = valencies[frame_IDs[j][i]]
            if N > 1: # if N=1, |psi| will trivially be 1
                psi_frame.append(abs(psi))
                r_psi[valency].append(abs(psi))
            r = sqrt((COM[0]-p[0])**2 + (COM[1]-p[1])**2)
            r_r[valency].append(r)

        psi_data.append(psi_frame)
        radial_psi.append(r_psi)
        radial_r.append(r_r)

    max_density = max([max(densities) for densities in frame_densities])
    frame_densities = [densities / max_density
                       for densities in frame_densities]

    #take averages
    radial_psi = take_avg(radial_psi, True)
    radial_densities = take_avg(radial_densities, True)
    radial_r = take_avg(radial_r, False)

    m = max([max(x) for x in radial_densities])
    radial_densities /= m
    np.savez(fname + "_DATA.npz", psi=psi_data, densities=frame_densities,
             radial_psi=radial_psi, radial_densities=radial_densities,
             radial_r=radial_r)
