#-------------------------------------------------------------------------------
# Name:        GAssist_Main.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Command line Main method for running the GAssist algorithm.  
#              Handles inputs, builds the environment, and runs the GAssist algorithm
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist import *
from GAssist_Environment import *
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
    10: Initialization - 'none', 'smart', 'cw'
    11: PopSize - '100, 500, 1000'
    12: Wild frequency - 0.5, 0.75
    13: Default Class Handeling - auto, 0, disabled  - automatic default class evolution, vs. pre-specify the default class (can also add a method which chooses via which class is most represented.
    14: MDL fitness - '0', '1' - where 0 is False, and 1 is True.
    15: Windows - '1', '2', '4' - number of partitions of the dataset used in the learning process. (higher values tend to promote generalization, and reduce computational time.)
"""
# *************************************************************************************************

graphPerformance = False # Built in graphing ability, currently not functional, but mechanism is in place. NOT MEANT TO BE USED ON CLUSTER.
numArgs = len(argv)
print "Arguments: " + str(numArgs)
if numArgs == 16:
    if argv[1] == 'gh': #Different rule representations could be programmed but have not been in this implementation.
        print ("Format Training data: "+argv[2]+"  using a "+argv[6]+" bit coding scheme.")
        
        #Sets up up algorithm to be run.
        e = GAssist_Environment(str(argv[2]), str(argv[3]), int(argv[6]), str(argv[10]))
        cons.setConstants(int(argv[11]), float(argv[12]), str(argv[13]), e, str(argv[14]), str(argv[15])) 
        sampleSize = e.getNrSamples()
        gassist = GAssist(e, argv[4], argv[5], int(argv[6]), int(argv[7]), graphPerformance)
        
        #Figure out the iteration stops for evaluation, and the max iterations.
        iterList = argv[9].split('.')
        for i in range(len(iterList)):
            iterList[i] = int(iterList[i])
        lastIter = iterList[len(iterList)-1]
        
        #Set some GAssist parameters.
        if argv[9] == 'Default':
            gassist.setTrackingIterations(sampleSize)
        else:
            gassist.setTrackingIterations(int(argv[8]))   
        gassist.setNumberOfTrials(lastIter, iterList)   
        gassist.setInitialization(str(argv[10])) #NEW - for intelligent initialization
        
        #Run the GAssist Algorithm 
        gassist.runGAssist()        
        
    else:
        print "There is no environment handling code for the specified environment"
else:
    print helpstr
    sys.exit()