#-------------------------------------------------------------------------------
# Name:        GAssist_Constants.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Specifies default GAssist run constants.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Environment import *

class CGAssistConstants:
    """ Specifies default GAssist run constants. """

    #Initialization Parameters
    agentInitWidth = 20 # Initial complexity of agent    
    dontCare = '#'
    
    #Major Architectural Parameters
    doRuleDeletion = False #was self.
    doHierarchicalSelection = False #was self.

    #Other Pressure Parameters  
    hierarchicalSelectionThreshold = 0
    iterationHierarchicalSelection = 24 
    iterationRuleDeletion = 5
    sizePenaltyMinRules = 4
    iterationMDL = 25 #the iteration where MDL kicks in.
    initialTheoryLengthRatio = 0.075 #used in calculating the theory weight
    weightRelaxFactor = 0.90 
    ruleDeletionMinRules = 12
    
    tournamentSize = 3 #in actuality the T-size is the value entered here + 1 according to the coding.
    probCrossover = 0.6 #probability of a crossover
    probMutationInd = 0.6 

    
    def setConstants(self, popSize, wild, defaultClass, env, MDL, windows):
        """ Initialize user defined constants. """
        self.popSize = popSize
        self.wild = wild
        self.numStrata = int(windows)
        if int(MDL) == 0:
            self.useMDL = False
        elif int(MDL) == 1:
            self.useMDL = True
        else:
            self.useMDL = -1
            print "Incorrect MDL setting selected."
            
        if defaultClass == "auto":
            self.defaultClass = str(defaultClass)
            self.nichingEnabled = True
            # Tracks last 15 average accuracies for each niche. (only needed with niching)
            self.accDefaultRules = [[] for i in range(env.getNrActions())] #hope this works. - nope it doesn't
        elif defaultClass == "disabled":
            self.defaultClass = str(defaultClass) 
            self.nichingEnabled = False
        else:
            self.defaultClass = int(defaultClass)
            self.nichingEnabled = False
    
    def setLearning(self, pol):
        """ Sets the learning progress each iteration. """
        self.percentageOfLearning = pol
    
    def setRuleDelete(self, delete):
        self.doRuleDeletion = delete

    def setHierarchy(self, hier):
        self.doHierarchicalSelection = hier
        
    def setDeleteMinRules(self, dmr):
        self.ruleDeletionMinRules = dmr
        
    def setNichingStatus(self, niche):
        self.nichingEnabled = niche

GAssistConstants = CGAssistConstants()
cons = GAssistConstants