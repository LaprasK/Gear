#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:34:28 2017

@author: zhejun
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def circle_click(im):
    """saves points as they are clicked, then find the circle that they define

    To use:
    when image is shown, click three non-co-linear points along the perimeter.
    neither should be vertically nor horizontally aligned (gives divide by zero)
    when three points have been clicked, a circle should appear.
    Then close the figure to allow the script to continue.
    """
    

    clicks = []
    center = []
    if isinstance(im, basestring):
        im = plt.imread(im)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)

    def circle_click_connector(click):
        """receive and save clicks. when there are three, calculate the circle

         * swap x, y to convert image coordinates to cartesian
         * the return value cannot be saved, so modify a mutable variable
           (center) from an outer scope
        """
        clicks.append([click.ydata, click.xdata])
        print 'click {}: x: {:.2f}, y: {:.2f}'.format(len(clicks), *clicks[-1])
        if len(clicks) == 3:
            center.extend(circle_three_points(clicks))
            print 'center {:.2f}, {:.2f}, radius {:.2f}'.format(*center)
            cpatch = matplotlib.patches.Circle(
                center[1::-1], center[2], linewidth=3, color='g', fill=False)
            ax.add_patch(cpatch)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', circle_click_connector)
    plt.show()
    return center


def circle_three_points(*xs):
    """ With three points, calculate circle
        e.g., see paulbourke.net/geometry/circlesphere
        returns center, radius as (xo, yo), r
    """
    xs = np.squeeze(xs)
    if xs.shape == (3, 2):
        xs = xs.T
    (x1, x2, x3), (y1, y2, y3) = xs

    ma = (y2-y1)/(x2-x1)
    mb = (y3-y2)/(x3-x2)
    xo = ma*mb*(y1-y3) + mb*(x1+x2) - ma*(x2+x3)
    xo /= 2*(mb-ma)
    yo = (y1+y2)/2 - (xo - (x1+x2)/2)/ma
    r = ((xo - x1)**2 + (yo - y1)**2)**0.5

    return xo, yo, r
