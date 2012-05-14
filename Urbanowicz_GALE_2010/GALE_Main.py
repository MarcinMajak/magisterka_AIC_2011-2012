#!/usr/bin/env python
# -*- coding: utf-8 -*-


#Import modules
from GALE import *
from GALE_Environment import *
import time
import sys
argv = sys.argv

# *************************************************************************************************
helpstr = """
Parameters:
    1:  Specify the environment - 'gh' - gh = genetic heterogeneity snp datasets with discretely coded genotype attributes and a case/control classes.
    2:  Training Dataset - '/SomePath/SomeTrainfile.txt'
    3:  Testing Dataset - '/SomePath/SomeTestfile.txt'
    4:  Output Filename (Learning Progress Tracking) - '/SomePath/SomeProgressName'
    5:  Output Filename (Final Pop Evaluation) - '/SomePath/SomeFinalPopName'
    6:  Coding Length - '1'
    7:  Dataset Partitions for CV - '10'
    8:  Performance Tracking Cycles - '1'
    9:  Learning iterations - '100.500.1000' or for testing '10.20'
    10: BoardSize (X-axis and Y-axis) - '10', or '25', or '50'
    11: Data Distribution - '0' or '1' or '2' or '3' = uniform, random, pyrimidal, random pyrimidal
    12: Pruning - '0' or '1'
    13: Wild frequency - 0.5, 0.75
"""
# *************************************************************************************************

graphPerformance = False # Built in graphing ability, currently not functional, but mechanism is in place. NOT MEANT TO BE USED ON CLUSTER.
numArgs = len(argv)
print "Arguments: " + str(numArgs)
if numArgs == 14:
    if argv[1] == 'gh': #Different rule representations could be programmed but have not been in this implementation.
        print ("Format Training data: "+argv[2]+"  using a "+argv[6]+" bit coding scheme.")
        
        #Sets up up algorithm to be run.
        GALEConstants.setConstants(int(argv[12]), float(argv[13]))
        e = GALE_Environment(str(argv[2]), str(argv[3]), int(argv[6]))
        sampleSize = e.getNrSamples()
        gale = GALE(e, argv[4], argv[5], int(argv[6]), int(argv[7]), graphPerformance, int(argv[10]), int(argv[10]), int(argv[11]))
        
        #Figure out the iteration stops for evaluation, and the max iterations.
        iterList = argv[9].split('.')
        for i in range(len(iterList)):
            iterList[i] = int(iterList[i])
        lastIter = iterList[len(iterList)-1]
        
        #Set some GALE parameters.
        if argv[9] == 'Default':
            gale.setTrackingIterations(sampleSize)
        else:
            gale.setTrackingIterations(int(argv[8]))   
        gale.setNumberOfTrials(lastIter, iterList)   
        
        #Run the GALE Algorithm 
        gale.runGALE()        
        
    else:
        print "There is no environment handling code for the specified environment"
else:
    print helpstr
    sys.exit()
