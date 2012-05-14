#-------------------------------------------------------------------------------
# Name:        GAssist_Global_MDL.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Handles MDL fitness calculations and controls variable activation of fitness calculations.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Constants import * 
from GAssist_Environment import *

class GAssistGlobalMDL:
    def __init__(self, env): 
        """ Initialize the global tracking MDL parameters. """
        self.theoryWeight = 0.0
        self.activated = False
        self.fixedWeight = False
        self.e = env
    
    def newIteration(self, iteration, pop):
        """ Called by Timers to alter how fitness is calculated over time. """
        if not cons.useMDL:
            return False
        
        ind = pop.getBest()
        
        updateWeight = False
        if iteration == cons.iterationMDL:
            print "Iteration " + str(iteration) + " MDL fitness activated"
            self.activated = True
            error = ind.getExceptionsLength() 
            theoryLength = ind.getTheoryLength()
            theoryLength *= self.e.getNrActions()
            theoryLength /= ind.agnPer.getNumAliveRules()
            
            self.theoryWeight = (cons.initialTheoryLengthRatio / (1.0 - cons.initialTheoryLengthRatio)) * (error / theoryLength)
            updateWeight = True
            
        if self.activated and not self.fixedWeight and pop.last10IterationsAccuracyAverage == 1.0:  
            print "Fixed Weight activated! = only if perfect accuracy has been obtained."
            self.fixedWeight = True

        if self.activated and not self.fixedWeight: 
            if ind.agnPer.getAccuracy() != 1.0:  
                if pop.getIterationsSinceBest() == 10: #theory weight updated every 10 iterations.
                    self.theoryWeight *= cons.weightRelaxFactor
                    print "New theory weight = " + str(self.theoryWeight)
                    updateWeight = True

        if updateWeight:
            pop.resetBestStats()
            return True
            
        return False
    

    def mdlFitness(self, ind):
        """ Computes the MDL based fitness - which places emphasis on a maximally small/simple and accurate set of rules as making up a most fit agent. """
        mdlFit = 0.0
        ind.computeTheoryLength() 
        if self.activated: 
            mdlFit = ind.getTheoryLength() * self.theoryWeight 
        exceptionsLength = 105.00 - ind.agnPer.getAccuracy() * 100.0 
        ind.setExceptionsLength(exceptionsLength) 
        mdlFit += exceptionsLength
        return mdlFit
    