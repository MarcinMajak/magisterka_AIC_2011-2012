#!/usr/bin/env python

class CGALEConstants:
    """ Specifies default GALE run constants. """

    #Initialization Parameters
    iAgent = 0.8    #initial fraction of agents filling board
    agentInitWidth = 12 # Initial complexity of agent    
    
    #Major Architectural Parameters
    neighborhood = 1  # r-value from GALE paper.  CHECK THIS!
    dontCare = '#'
        
    #Other Pressure Parameters  
    MergeP = 0.4
    kThreshold = -0.25 #parameter that controls the survival pressure over the current cell 
    P_pickLargerX = 0.5  # probability that after crossover the new agent will be the bigger one
    SomaticMutationP = 0.01 #Probability that a point will be mutated.
    accuracyScaleF = 2 #should be 2  # power to which accuracy is multiplied for scaling accuracy
    MaxSplitP = 0.5 #Also 0.01 - Split probability - may be based on fitness as well 
    
    
    def setConstants(self, prune, wild):
        """ Sets whether pruning is turned on, and the rate at which wilds are incorporated durring mutations. """
        if prune == 0:
            self.Prune = False
        else:
            self.Prune = True
        self.Wild = wild
    
    
GALEConstants = CGALEConstants()
cons = GALEConstants    #To access one of the above constant values from another module, import GALE_Constants * and use "cons.Xconstant"
