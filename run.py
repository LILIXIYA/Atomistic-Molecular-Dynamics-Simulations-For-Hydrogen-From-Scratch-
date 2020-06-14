# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 04:46:46 2019

@author: LXS
"""


import makeSystem
import writeXYZ
import forces
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

class Simulation:
    
    def __init__(self, numberOfMolecules, percentageEthanol, boxSize, cutoff):
            
        self.positions, self.velocity, self.atomNames, self.atomType, self.weights, self.bonds, \
        self.bondParameters, self.angles, self.angleParameters, self.dihedrals, \
        self.dihedralParameters, self.sigma, self.epsilon, self.molecules = \
        makeSystem.makeSystem(numberOfMolecules, percentageEthanol, boxSize)
        
        self.numberOfAtoms = self.positions.shape[0]
        self.boxSize = boxSize
        self.cutoff = cutoff
    
    
    def integrate(self, dT, endTime, thermostat, constTemp):
        potentialEnergy = []
        kineticEnergy = []
        temperature = []
        
        
        t = 0.0
        index = 0
        enIndex = 0
        
        self.thermostat(True, constTemp)
        
        while t < endTime:
            index = index + 1
            enIndex = enIndex + 1
            if index > 9:
                index = 0
                print("Percentage :", 100.0*t/endTime)
            
            kin, pot = forces.calculateEnergy(self.positions, self.velocity, self.atomNames, self.atomType, self.weights, self.bonds, \
            self.bondParameters, self.angles, self.angleParameters, self.dihedrals, \
            self.dihedralParameters, self.sigma, self.epsilon, self.molecules, self.cutoff, self.boxSize)
            
            
            potentialEnergy.append(pot)
            kineticEnergy.append(kin)
            
            t = t + dT
            
            temp = self.thermostat(thermostat, constTemp)
            
            temperature.append(temp)
            
            self.positions, self.velocity = self.velocityVerlet(dT)
        
        kineticEnergy = np.array(kineticEnergy)
        kineticEnergy *= 10**4
        potentialEnergy = np.array(potentialEnergy)
        
        plt.subplot(2,1,1)
        plt.plot(kineticEnergy,'r-',potentialEnergy,'g-',kineticEnergy+potentialEnergy,'b')
        plt.title('Energy')
        plt.xlabel('fs')
        plt.ylabel('kJ/mol')
        plt.legend(["Total energy", "Potential energy", "Kinetic energy"])
        
        plt.subplot(2,1,2)
        plt.plot(temperature)
        plt.title('Temperature')
        plt.xlabel('fs')
        plt.ylabel('K')

        self.writeToXYZ('test.xyz',0)
        
    
    
    def writeToXYZ(self, fileName, time):
        writeXYZ.deleteFile(fileName)
        writeXYZ.writeFile(fileName, self.numberOfAtoms, self.atomNames,
                           self.positions, time)
    
    
    def thermostat(self,thermostat, constTemp):
        const = 8.314459865590528e-07
        
        aveVel = 0.0
        for i in range(self.numberOfAtoms):
            aveVel += self.weights[i]*np.linalg.norm(self.velocity[i])**2
        
        temperature = aveVel/(3*self.numberOfAtoms*const)
        
        if thermostat:
            scale = (constTemp/temperature)**.5
            
            self.velocity = scale*self.velocity
        
        return temperature
    
    def velocityVerlet(self,dT):
        
        
        F1 = forces.calculateForces(self.positions, self.velocity, self.atomNames, self.atomType, self.weights, self.bonds, \
        self.bondParameters, self.angles, self.angleParameters, self.dihedrals, \
        self.dihedralParameters, self.sigma, self.epsilon, self.molecules, self.cutoff, self.boxSize)
        
        
        positionsNew = self.positions + dT*self.velocity + .5*(dT**2)*(10**-4)*np.divide(F1,self.weights[:,np.newaxis])
        
        
        F2 = forces.calculateForces(positionsNew, self.velocity, self.atomNames, self.atomType, self.weights, self.bonds, \
        self.bondParameters, self.angles, self.angleParameters, self.dihedrals, \
        self.dihedralParameters, self.sigma, self.epsilon, self.molecules, self.cutoff, self.boxSize)
        
        velocityNew = self.velocity + .5*dT*(10**-4)*np.divide(F1+F2,self.weights[:,np.newaxis])
        
        return positionsNew, velocityNew
        
        
