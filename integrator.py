# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 04:56:24 2019

@author: LXS
"""

import numpy as np

def velocityVerlet(positions, momentum, weights, force, dT):
    
    F0 = force.calculateForces(positions)
    
    positionNew = positions + dT*momentum + \
    .5*(dT**2)*np.divide(F0,weights)
    
    F1 = force.calculateForces(positionNew)
    
    velocityNew = momentum + .5*dT*np.divide(np.add(F0,F1),weights)
    
    
    return positionNew, velocityNew


def Euler(positions, momentum, weights, force, dT):
    F0 = force.calculateForces(positions)
    
    positionNew = positions + dT*momentum + \
    .5*(dT**2)*np.divide(F0,weights)
    
    velocityNew = momentum + dT*np.divide(F0,weights)
    
    return positionNew, velocityNew