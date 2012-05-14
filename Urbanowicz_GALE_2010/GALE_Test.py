#!/usr/bin/env python

#Import modules
from GALE import *
from GALE_Environment import *
from GALE_Constants import *  

def main():
    """ Runs independently from command line to test the GALE algorithm. """
    graphPerformance = False # Built in graphing ability, currently not functional, but mechanism is in place.
    trainData = "2_1000_0_1600_0_0_CV_0_Train.txt"
    testData = "2_1000_0_1600_0_0_CV_0_Test.txt"
    outProg = "GH_GALE_ProgressTrack"
    outPop = "GH_GALE_PopulationOut"
    bitLength = 1    # This implementation is not yet set up to handle other rule representations, or bit encoding lengths.
    CVpartitions = 10
    trackCycles = 1
    
    iterInput = '5.10.20'  
    xdim = 10
    ydim = 10
    dist = 2
    wild = 0.75
    prune = 1
    
    #Figure out the iteration stops for evaluation, and the max iterations.
    iterList = iterInput.split('.')
    for i in range(len(iterList)):
        iterList[i] = int(iterList[i])
    lastIter = iterList[len(iterList)-1]  

    #Sets up up algorithm to be run.
    GALEConstants.setConstants(prune, wild)
    e = GALE_Environment(trainData,testData,bitLength)
    sampleSize = e.getNrSamples()
    gale = GALE(e, outProg, outPop, bitLength, CVpartitions, graphPerformance, xdim, ydim, dist)
    
    #Set some GALE parameters.
    if trackCycles == 'Default':
        gale.setTrackingIterations(sampleSize)
    else:
        gale.setTrackingIterations(trackCycles) 
    gale.setNumberOfTrials(lastIter, iterList)  
    
    #Run the GALE Algorithm 
    gale.runGALE()
     
if __name__ == '__main__':
    main()
