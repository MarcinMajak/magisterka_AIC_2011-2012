#-------------------------------------------------------------------------------
# Name:        GALE_Test.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: A testing substitute for GAssist_Main.py to run GAssist from Eclipse(with Pydev).
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist import *
from GAssist_Environment import *
from GAssist_Constants import * 

def main():
    """ Runs independently from command line to test the GAssist algorithm. """
    graphPerformance = False # Built in graphing ability, currently not functional, but mechanism is in place.
    trainData = "2_1000_0_1600_0_0_CV_0_Train.txt"
    testData = "2_1000_0_1600_0_0_CV_0_Test.txt"
    outProg = "GH_GAssist_ProgressTrack"
    outPop = "GH_GAssist_PopulationOut"
    bitLength = 1    # This implementation is not yet set up to handle other rule representations, or bit encoding lengths.
    CVpartitions = 10
    trackCycles = 1
    
    # Run Parameters - User specified.
    iterInput = '20.50.100' 
    pop = 100
    wild = 0.5
    defaultClass = "0"  #auto, 0, disabled   
    init = "cw"  #'none', 'smart', 'cw'
    MDL = 1
    windows = 2
    
    #Figure out the iteration stops for evaluation, and the max iterations.
    iterList = iterInput.split('.')
    for i in range(len(iterList)):
        iterList[i] = int(iterList[i])
    lastIter = iterList[len(iterList)-1]  

    #Sets up up algorithm to be run.
    e = GAssist_Environment(trainData,testData,bitLength, init) 
    cons.setConstants(pop, wild, defaultClass, e, MDL, windows) 
    sampleSize = e.getNrSamples()
    gassist = GAssist(e, outProg, outPop, bitLength, CVpartitions, graphPerformance) 
    
    #Set some GAssist parameters.
    if trackCycles == 'Default':
        gassist.setTrackingIterations(sampleSize)
    else:
        gassist.setTrackingIterations(trackCycles) 
    gassist.setNumberOfTrials(lastIter, iterList)  
    gassist.setInitialization(init)
    #Run the GAssist Algorithm 
    gassist.runGAssist()
     
     
if __name__ == '__main__':
    main()