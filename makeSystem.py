# -*- coding: utf-8 -*-
"""
This file contains the routine to obtain an initial system of a mixture of
water and ethanol molecules.

A list of bonds, angles, etc is computed in this routine and returned.

Furthermore, a routine for setting the initial velocity is provided.

This file is not optimized, as it relies on Python functions not supported by
Numba nopython mode, and as it is only called once, this does not have a big
impact on the running time of the simulation.
"""

import numpy as np

# Water parameters, and base coordinates
waterStructure = ('O','H','H')
baseWaterCoordinates = np.array([[0.0, 0.0, 0.0],
                                 [0.6, 0.4, 0.6], 
                                 [0.5, -0.5, -0.6]])

waterWeights = [15.994, 1.0080, 1.0080]

bondForcesWater = [[np.array([1,2],dtype='int'),5024.16,0.9572],
                    [np.array([1,3],dtype='int'),5024.16,0.9572]
                    ]

angleForcesWater = [[np.array([2,1,3],dtype='int'),628.02,104.52]]

dihedralForcesWater = None

parametersLennardJonesWater = [[np.array([3.15061, 0.66386])],
                                   [np.array([0.0, 0.0])],
                                   [np.array([0.0, 0.0])]
                                   ]

# Ethanol parameters, and base coordinates
ethanolStructure = ('C','H','H','H','C','H','H','O','H')

baseEthanolCoordinates = np.array([[0.0, 0.0, 0.0], [-0.006, -1.115, 0.017], [0.0512, 0.349, 0.927], [-1.055,  0.356,  0.033], [ 0.715,  0.515, -1.251], [1.777,  0.179, -1.284], [0.193,  0.204, -2.186], [0.73,   1.918, -1.226], [-0.159,  2.211, -1.334]])

ethanolWeights = [12.0110,1.0080,1.0080,1.0080,12.0110,1.0080,1.0080,15.9994,1.0080]

bondForcesEthanol = [[np.array([1,5],dtype='int'),2242.624,1.529],
                      [np.array([1,3],dtype='int'),2845.12,1.090],
                      [np.array([1,4],dtype='int'),2845.12,1.090],
                      [np.array([1,2],dtype='int'),2845.12,1.090],
                      [np.array([6,5],dtype='int'),2845.12,1.090],
                      [np.array([7,5],dtype='int'),2845.12,1.090],
                      [np.array([8,5],dtype='int'),2677.76,1.410],
                      [np.array([9,8],dtype='int'),4627.50,0.945]]

angleForcesEthanol = [[np.array([2,1,5],dtype='int'),292.880,108.5],
                       [np.array([3,1,5],dtype='int'),292.880,108.5],
                       [np.array([4,1,5],dtype='int'),292.880,108.5],
                       
                       [np.array([4,1,3],dtype='int'),276.144,107.8],
                       [np.array([4,1,2],dtype='int'),276.144,107.8],
                       [np.array([3,1,2],dtype='int'),276.144,107.8],
                       [np.array([6,5,7],dtype='int'),276.144,107.8],
                       
                       [np.array([1,5,7],dtype='int'),313.800,110.7],
                       [np.array([1,5,6],dtype='int'),313.800,110.7],
                       
                       [np.array([1,5,8],dtype='int'),414.400,109.5],
                       
                       [np.array([5,8,9],dtype='int'),460.240,108.5],
                       
                       [np.array([6,5,8],dtype='int'),292.880,109.5],
                       [np.array([7,5,8],dtype='int'),292.880,109.5]
                       ]

dihedralForcesEthanol = [[np.array([2,1,5,6],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         [np.array([3,1,5,6],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         [np.array([4,1,5,6],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         [np.array([2,1,5,7],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         [np.array([3,1,5,7],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         [np.array([4,1,5,7],dtype='int'), np.array([0.62760, 1.88280, 0.00000, -3.91622])],
                         
                         [np.array([2,1,5,8],dtype='int'), np.array([0.97905, 2.93716, 0.00000, -3.91622])],
                         [np.array([3,1,5,8],dtype='int'), np.array([0.97905, 2.93716, 0.00000, -3.91622])],
                         [np.array([4,1,5,8],dtype='int'), np.array([0.97905, 2.93716, 0.00000, -3.91622])],
                         
                         [np.array([1,5,8,9],dtype='int'), np.array([-0.44310, 3.83255, 0.72801, -4.11705])],
                         
                         [np.array([6,5,8,9],dtype='int'), np.array([0.94140, 2.82420, 0.00000, -3.76560])],
                         [np.array([7,5,8,9],dtype='int'), np.array([0.94140, 2.82420, 0.00000, -3.76560])]
                        ]

parametersLennardJonesEthanol = [[np.array([3.5, 0.276144])],
                                  [np.array([2.5, 0.125520])],
                                  [np.array([2.5, 0.125520])],
                                  [np.array([2.5, 0.125520])],
                                  [np.array([3.5, 0.276144])],
                                  [np.array([2.5, 0.125520])],
                                  [np.array([2.5, 0.125520])],
                                  [np.array([3.12, 0.711280])],
                                  [np.array([0.0, 0.0])]
                                  ]

 

def makeSystem(numberOfMolecules, percentageEthanol, boxSize):
    
    # Determine how many ethanol molecules are there, and set the molecules i
    # in ethanolIndex to 1, indicating it is ethanol
    numberOfEthanolMolecules = int(round(percentageEthanol*numberOfMolecules))
    if numberOfEthanolMolecules > numberOfMolecules:
        numberOfEthanolMolecules = numberOfMolecules
    ethanolIndices = np.random.choice(numberOfMolecules,
                                      numberOfEthanolMolecules,
                                      replace = False)
    ethanolIndex = np.zeros(numberOfMolecules)
    ethanolIndex[ethanolIndices] = 1
    
    
    numberOfAtoms = 3*(numberOfMolecules - numberOfEthanolMolecules) \
    + 9*numberOfEthanolMolecules
    
    bonds = []
    bondParameters = []
    angles = []
    angleParameters = []
    dihedrals = []
    dihedralParameters = []
    weights = []
    molecules = []
    atomType = [] # Array to see to which kind of molecule an atom belongs,
                  # 0 for water,1 for ethanol
    atomNames = []
    
    index = 0
    
    for i in range(numberOfMolecules):
        if ethanolIndex[i] < .5:
            # Molecule i is a water molecule
            molecules.append(3*[i])
            
            atomType.append(3*[0])
            
            atomNames.append(waterStructure)
            
            for i in waterWeights:
                weights.append(i)
            
            # Add bond parameters
            for i in bondForcesWater:
                bonds.append(i[0] - 1 + index)
                bondParameters.append(np.array([i[1],i[2]]))
            
            # Add angle parameters
            for i in angleForcesWater:
                angles.append(i[0] - 1 + index)
                angleParameters.append(np.array([i[1],i[2]*np.pi/180]))
            
            # Update the index
            index += 3
            
        else:
            # Molecule i is an ethanol molecule
            molecules.append(9*[i])
            
            atomType.append(9*[1])
            
            atomNames.append(ethanolStructure)
            
            for i in ethanolWeights:
                weights.append(i)
            
            # Add bond parameters
            for i in bondForcesEthanol:
                bonds.append(i[0] - 1 + index)
                bondParameters.append(np.array([i[1],i[2]]))
            
            # Add angle parameters
            for i in angleForcesEthanol:
                angles.append(i[0] - 1 + index)
                angleParameters.append(np.array([i[1],i[2]*np.pi/180]))
            
            # Update dihedral index, as there are no dihedrals for water
            for i in dihedralForcesEthanol:
                dihedrals.append(i[0] - 1 + index)
                dihedralParameters.append(i[1])
            
            # Update the index
            index += 9
            
    
    
    # flatten the sublists 
    molecules = [item for sublist in molecules for item in sublist]
    atomType = [item for sublist in atomType for item in sublist]
    atomNames = [item for sublist in atomNames for item in sublist]
    
    # Convert all lists to np.arrays
    molecules = np.array(molecules)
    atomType = np.array(atomType)
    
    weights = np.array(weights)
    
    bonds = np.array(bonds)
    bondParameters = np.array(bondParameters)
    
    angles = np.array(angles)
    angleParameters = np.array(angleParameters)
    
    dihedrals = np.array(dihedrals)
    dihedralParameters = np.array(dihedralParameters)
    
    # Make the Lennard Jones parameters
    epsilon = []
    sigma = []
    for i in range(numberOfMolecules):
        if ethanolIndex[i] < 0.5:
            for j in range(3):
                sigma.append(parametersLennardJonesWater[j][0][0])
                epsilon.append(parametersLennardJonesWater[j][0][1])
        else:
            for j in range(9):
                sigma.append(parametersLennardJonesEthanol[j][0][0])
                epsilon.append(parametersLennardJonesEthanol[j][0][1])
    sigma = np.array(sigma)
    epsilon = np.array(epsilon)
    
    sigma = sigma + sigma[:,np.newaxis]
    sigma = sigma*.5
    
    epsilon = epsilon*epsilon[:,np.newaxis]
    epsilon = np.sqrt(epsilon)
    
    
    # Assign the positions of the molecules:
    dim = int(round(numberOfMolecules**(.333333333)))
    
    positions = []
    
    index = 0
    for x in range(dim):
        for y in range(dim):
            for z in range(dim):
                base = np.array([x+1,y+1,z+1])/(dim+1.0)
                base = base * boxSize
                
                if ethanolIndex[index] < 0.5:
                    for i in range(3):
                        positions.append(base + baseWaterCoordinates[i])
                else:
                    for i in range(9):
                        positions.append(base + baseEthanolCoordinates[i])
                
                index = index + 1
    positions = np.array(positions)
    
    # Assign velocities of the molecules, we will use an uniform distribution
    # over [-1,1], and scale to zero linear momentum
    # Velocity is in Amstrong/fs
    
    velocity = []
    
    for i in range(numberOfMolecules):
        vel = np.random.uniform(-1.0,1.0,3)
        if ethanolIndex[i] < 0.5:
            for j in range(3):
                velocity.append(vel)
        else:
            for j in range(9):
                velocity.append(vel)
    velocity = np.array(velocity)
    
    velScale = np.sum(velocity,axis=0)
    velScale = velScale/float(numberOfAtoms)
    
    velocity = velocity - velScale
    
    return positions, velocity, atomNames, atomType, weights, bonds, bondParameters, \
    angles, angleParameters, dihedrals, dihedralParameters, sigma, epsilon, molecules