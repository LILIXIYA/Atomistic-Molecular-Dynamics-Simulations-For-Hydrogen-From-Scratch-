# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:17:50 2019

@author: LXS
"""

import numpy as np

waterStructure = ('O','H','H')
baseWaterCoordinates = np.array([[0.0, 0.0, 0.0],
                                 [0.6, 0.4, 0.6], 
                                 [0.5, -0.5, -0.6]])
baseEthanolCoordinates = (np.array([0.99399,0.07994,0.05605]),
      np.array([0.60342,0.05616,-0.96663]),
      np.array([0.60759,-0.78611,0.60064]),
      np.array([0.60862,0.99235,0.52285]),
      np.array([2.50855,0.08277,0.05125]),
      np.array([2.89445,0.12202,1.07403]),
      np.array([2.90120 ,-0.81732,-0.43143]),
      np.array([2.98949,1.22165,-0.64781]),
      np.array([3.88949,1.22165,-0.64781])
      )
baseEthanolCoordinates = baseEthanolCoordinates - baseEthanolCoordinates[0]


t = np.linalg.norm(baseEthanolCoordinates[3])


 

def makeFile(numberOfMolecules, percentageEthanol, boxSize):
    pass