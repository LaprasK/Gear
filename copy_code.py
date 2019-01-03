#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 14:51:56 2017

@author: zhejun
"""

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


        
                          