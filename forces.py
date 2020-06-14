# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 11:07:34 2019

@author: LXS
"""

import numpy as np
from numba import jit

def calculateEnergy(positions, velocity, atomNames, atomType, weights, bonds, bondParameters, \
    angles, angleParameters, dihedrals, dihedralParameters, sigma, epsilon, molecules, cutoff, boxSize):
    kineticEnergy = 0.0
    potentialEnergy = 0.0
    
    kineticEnergy = calculateKineticEnergy(velocity, weights)
    
    potentialEnergy += calculateBondEnergy(positions, bonds, bondParameters)
    potentialEnergy += calculateAngleEnergy(positions, angles, angleParameters)
    if dihedrals.shape[0] > 0:
        potentialEnergy += calculateDihedralEnergy(positions, dihedrals, dihedralParameters)
    
    potentialEnergy += calculateLennardJonesEnergy(positions, molecules, epsilon, sigma, cutoff, boxSize)
    
    return kineticEnergy, potentialEnergy

def calculateForces(positions, velocity, atomNames, atomType, weights, bonds, bondParameters, \
    angles, angleParameters, dihedrals, dihedralParameters, sigma, epsilon, molecules, cutoff, boxSize):
    
    forces = np.zeros(positions.shape)
    
    forces += calculateBondForce(positions, bonds, bondParameters)
    
    forces += calculateAngleForce(positions, angles, angleParameters)
    
    if dihedrals.shape[0] > 0:
        forces += calculateDihedralForce(positions, dihedrals, dihedralParameters)
        
    forces += calculateLJForce(positions, molecules, epsilon, sigma, cutoff, boxSize)
    
    return forces
    
@jit(nopython=True)
def calculateBondForce(positions, bonds, bondParameters):
    forces = np.zeros(positions.shape)
    for i in range(bonds.shape[0]):
        i1 = bonds[i][0]
        i2 = bonds[i][1]
        
        r = distance(positions[i1],positions[i2])
        v = positions[i1] - positions[i2]
        v = v/length(v)
        
        force =  -bondParameters[i][0] * (r-bondParameters[i][1]) * v
        forces[i1,:] = forces[i1,:] + force
        forces[i2,:] = forces[i2,:] - force
    return forces

@jit(nopython=True)
def calculateAngleForce(positions, angles, angleParameters):
    forces = np.zeros(positions.shape)
    for i in range(angles.shape[0]):
        i1 = angles[i][0]
        i2 = angles[i][1]
        i3 = angles[i][2]
        
        x1 = positions[i1]
        x2 = positions[i2]
        x3 = positions[i3]
        
        theta0 = angleParameters[i][1]
        k = angleParameters[i][0]
        
    
        ba = x1 - x2
        bc = x3 - x2
        
        theta = angle(ba,bc)
        if theta < 0:
            theta = -theta
        
        pa = cross(ba,bc)
        pa = cross(ba,pa)
        pa = pa/length(pa)
        fa = -k*(theta-theta0)*pa/length(ba)
        
        pc = cross(ba,bc)
        pc = cross(-bc,pc)
        pc = pc/length(pc)
        fc = -k*(theta-theta0)*pc/length(bc)
        
        fb = -fa-fc
        
        forces[i1] += fa
        forces[i2] += fb
        forces[i3] += fc
    
    return forces

@jit(nopython=True)
def calculateDihedralForce(positions, dihedrals,dihedralParameters):
    forces = np.zeros(positions.shape)
    for ind in range(dihedrals.shape[0]):
        
        i1 = dihedrals[ind][0]
        a = positions[i1]
        i2 = dihedrals[ind][1]
        b = positions[i2]
        i3 = dihedrals[ind][2]
        c = positions[i3]
        i4 = dihedrals[ind][3]
        d = positions[i4]
        
        p1 = cross(a-b,c-b)
        p2 = cross(d-c,b-c)
        
        theta1 = angle(a-b,c-b)
        theta2 = angle(b-c,d-c)
            
        theta = angle(p1,p2)
        theta = theta - np.pi
        
        C1 = dihedralParameters[ind][0]
        C2 = dihedralParameters[ind][1]
        C3 = dihedralParameters[ind][2]
        C4 = dihedralParameters[ind][3]
        
        const = .5*(C1*np.sin(theta) - 2*C2*np.sin(2*theta) + 3*C3*np.sin(3*theta) - 4*C4*np.sin(4*theta))
        
        fa = const*p1/(length(b-a)*np.sin(theta1))
        
        fd = const*p2/(length(d-c)*np.sin(theta2))
        
        o = (b+c)/2
        
        tc = -(cross(c-o,fd) + .5*cross(d-c,fd)+.5*cross(b-a,fa))
        
        fc = cross(tc,c-o)/(length(c-o)**2)
        
        fb = -fa-fc-fd
#        
        forces[i1] += fa
        forces[i2] += fb
        forces[i3] += fc
        forces[i4] += fd
        
    return forces

@jit(nopython=True)
def calculateLJForce(positions, molecules, epsilonMat, sigma, cutoff, boxSize):
    forces = np.zeros(positions.shape)
    
    totalAtoms = forces.shape[0]
    
    
    for i in range(totalAtoms):
        
        for j in range(i+1,totalAtoms):
            
            if molecules[i] is not molecules[j] and epsilonMat[i,j] > 10**-6:
                a = positions[i]
                b = np.copy(positions[j])
                
                for ind in range(3):
                    while b[ind] > a[ind] + boxSize/2.0:
                        b[ind] -= boxSize
                    while b[ind] < a[ind] - boxSize/2.0:
                        b[ind] += boxSize
                
                ba = a - b
                
                dist = length(ba)
                
#                dist = dist - boxSize*np.floor(dist/boxSize + 0.5)
                
                if dist < cutoff and dist > 1E-2:
                    ba = ba/dist
                    
                    eps = epsilonMat[i,j]
                    sig = sigma[i,j]
                    
                    sr = sig/dist
                    forces[i] += 24*eps*(2*(sr**12)-sr**6)*ba/dist
                    forces[j] -= 24*eps*(2*(sr**12)-sr**6)*ba/dist
                
    return forces

@jit(nopython=True)
def calculateBondEnergy(positions, bonds, bondParameters):
    energy = 0.0
    
    for i in range(bonds.shape[0]):
        i1 = bonds[i][0]
        i2 = bonds[i][1]
        
        r = distance(positions[i1],positions[i2])
        
        force = .5*bondParameters[i][0] * (r-bondParameters[i][1])**2
        
        energy = energy + force
    return energy

@jit(nopython=True)
def calculateAngleEnergy(positions, angles, angleParameters):
    energy = 0.0
    for i in range(angles.shape[0]):
        i1 = angles[i][0]
        i2 = angles[i][1]
        i3 = angles[i][2]
        
        x1 = positions[i1]
        x2 = positions[i2]
        x3 = positions[i3]
        
        theta0 = angleParameters[i][1]
        
        k = angleParameters[i][0]
        
        ba = x1 - x2
        bc = x3 - x2
        
        theta = angle(ba,bc)
        if theta < 0:
            theta = -theta
        energy = energy + .5*k*(theta-theta0)**2
    return energy

#@jit(nopython=True)
def calculateDihedralEnergy(positions, dihedrals, dihedralParameters):
    energy = 0.0
    
    for ind in range(dihedrals.shape[0]):
        i1 = dihedrals[ind][0]
        a = positions[i1]
        i2 = dihedrals[ind][1]
        b = positions[i2]
        i3 = dihedrals[ind][2]
        c = positions[i3]
        i4 = dihedrals[ind][3]
        d = positions[i4]
        
        p1 = cross(a-b,c-b)
        p2 = cross(d-c,b-c)
            
        theta = angle(p1,p2)
        theta = theta - np.pi
        
        C1 = dihedralParameters[ind][0]
        C2 = dihedralParameters[ind][1]
        C3 = dihedralParameters[ind][2]
        C4 = dihedralParameters[ind][3]
        
        const = .5*(C1*np.cos(theta) - C2*np.cos(2*theta) + C3*np.cos(3*theta) - C4*np.cos(4*theta))
        const += .5*(C1+C2+C3+C4)
        
        energy += const
    return energy

@jit(nopython=True)
def calculateLennardJonesEnergy(positions, molecules, epsilon, sigma, cutoff,boxSize):
    energy = 0.0
    
    totalAtoms = positions.shape[0]
    
    
    for i in range(totalAtoms):
        
        for j in range(i+1,totalAtoms):
            
            if molecules[i] is not molecules[j] and epsilon[i,j] > 10**-6:
                a = positions[i]
                b = positions[j]
                
                a = positions[i]
                b = np.copy(positions[j])
                
                for ind in range(3):
                    while b[ind] > a[ind] + boxSize/2.0:
                        b[ind] -= boxSize
                    while b[ind] < a[ind] - boxSize/2.0:
                        b[ind] += boxSize
                
                ba = a - b
                
                dist = length(ba)
                
                
                if dist < cutoff and dist > 10**-2:
                    
                    eps = epsilon[i,j]
                    sig = sigma[i,j]
                    
                    sr = sig/dist
                    energy += 4*eps*((sr**12)-sr**6)
                
    return energy  
                
                
        
@jit(nopython=True)
def calculateKineticEnergy(velocities, weights):
    energy = 0.0
    
    for i in range(velocities.shape[0]):
        energy = energy + .5*weights[i]*length(velocities[i])**2
        
    return energy

'''
Calculate the angle made between the vectors x1-x2 and x3-x2
'''
@jit(nopython=True)
def angle(x1,x2):

    theta = np.dot(x1,x2)
    l1 = length(x1)
    l2 = length(x2)
    
    if l1 is not 0 or l2 is not 0:
        theta = theta/(l1*l2)
    else:
        theta = 0
    
    if np.isnan(theta):
        print('nan detected in function angle(x1,x2)')
        print(x1, x2)
        print(l1, l2)
        print(theta)
        raise ZeroDivisionError()
        
    theta = np.arccos(theta)
    
    return theta

@jit(nopython=True)
def dot(x1,x2):
    return x1[0]*x2[0]+x1[1]*x2[1]+x1[2]*x2[2]


'''
Calculate the normalized vector in the plane made by x,y, orthogonal to x
'''
@jit(nopython=True)
def orthogonal(x,y):
    vec = cross(x,y)
    vec = -cross(x,vec)
    l = length(vec)
    vec = vec/l
    
    return vec

'''
Calculate the cross product of two vectors.
'''
@jit(nopython=True)
def cross(x,y):
    z0 = x[1]*y[2]-x[2]*y[1]
    z1 = -(x[0]*y[2]-x[2]*y[0])
    z2 = x[0]*y[1]-x[1]*y[0]
    
    return np.array([z0,z1,z2])
    
@jit(nopython=True)
def distance(x1,x2):
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    dz = x1[2] - x2[2]
    
    r = (dx*dx + dy*dy + dz*dz)**.5
    
    return r

@jit(nopython=True)
def length(x):
    dx = x[0]
    dy = x[1]
    dz = x[2]
    
    r = (dx*dx+dy*dy+dz*dz)**.5
    
    return r

    