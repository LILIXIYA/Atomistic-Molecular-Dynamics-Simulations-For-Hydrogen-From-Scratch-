# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 16:02:45 2019

@author: LXS
"""

import run
import time

#####################################################
# Simulation parameters

integrator = 'VelocityVerlet'
timeStep = 2 # Time step for the simulation, in femtosecond
totalTime = 100 # Total time for the simulation, in femtosecond

boxSize = 30.0 # dimension of the box, in Angstrom = A = 0.1 nM
             # Angstrom is chosen because this makes it easier for showing 
             # molecules in VMD, so no converting issues

outputXYZ = 'output.XYZ' # Output for the positions of the molecules,
                         # in a XYZ file


numberOfMolecules = 5**3

percentageEthanol = 000.0

cutoff = 8.0

thermostat = True # We will use the velocity scaling thermostat
constTemp = 300

t1 = time.time()

##################################################
# Start computation

simulation = run.Simulation(numberOfMolecules, percentageEthanol, boxSize, cutoff)

simulation.integrate(timeStep, totalTime, thermostat, constTemp)


t2 = time.time()
print('This computation took', t2-t1, ' seconds.')
print('This is ', (t2-t1)/(totalTime/timeStep),' second per computional time step')