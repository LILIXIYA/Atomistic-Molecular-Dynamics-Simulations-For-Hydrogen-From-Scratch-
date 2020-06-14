# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 04:47:17 2019

@author: LXS
"""

import os

def writeFile(file,numberOfAtoms, molecules,positions,time):
    fileExist = os.path.isfile(file)
    
    if not fileExist:
        myFile = open(file, 'w')
        print("File {} does not exist, creating a new file.".format(file))
    else:
        myFile = open(file,'a')
        
    myFile.write('{}\n'.format(numberOfAtoms))
    myFile.write("SOL t=\t {}\n".format(time))
    
    for i in range(numberOfAtoms):
        myFile.write("{}\t{}\t{}\t{}\n".format(molecules[i],\
                     positions[i][0],\
                     positions[i][1],\
                     positions[i][2]))
    
    myFile.close()

def deleteFile(file):
    '''
    Delete a file, useful for repeated test simulations.
    '''
    
    fileExist = os.path.isfile(file)
    
    if fileExist:
        os.remove(file)