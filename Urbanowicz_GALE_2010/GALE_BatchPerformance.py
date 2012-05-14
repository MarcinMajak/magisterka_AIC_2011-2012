#!/usr/bin/env python

#Import modules
from GALE_Constants import * 
import math

class GALEBatchPerformance:
    def __init__(self, nClass): 
        """ Initialize performance objects for a single agent. """
        self.iConfusionMatrix = []   #Confusion Matrix for tracking performance
        self.iDontKnow = 0  #Number of unclassified instances - (no match i believe)
        self.iTotal = 0     #Number of instances
        self.numClass = nClass  #Number of classes
        
        # Construct Confusion Matrix
        for i in range(self.numClass):
            self.iConfusionMatrix.append([])
            for j in range(self.numClass):
                self.iConfusionMatrix[i].append(0)
   
   
    def getConfusionMatrix(self):
        """ Return the confusion matrix. """
        return self.iConfusionMatrix
    
    
    def getDontKnow(self):
        """ Return the number of instances that could not be matched by any rule in the agent: (unclassified instances). """
        return self.iDontKnow
    
    
    def getTotal(self):
        """ Return the number of instances that were attempted to be matched by rules in the agent: (all instances). """
        return self.iTotal
 
 
    def getNoMatch(self):
        """ Returns the fraction of instances which could not be matched by the agent's rule set. """  
        if self.iTotal > 0:
            return self.iDontKnow/float(self.iTotal)
        return 0
    
    
    def getAccuracy(self):
        """ Returns the fraction of instances which were matched and correctly classified by the agent's rule set. """
        acc = 0
        for i in range(self.numClass):
            acc += self.iConfusionMatrix[i][i]
        if self.iTotal > 0:
            return acc/float(self.iTotal)
        return 0
    

    def getScaledAccuracy(self):
        """ Return the scaled accuracy. """
        tmpAcc = self.getAccuracy()
        return pow(tmpAcc,cons.accuracyScaleF)
    
    
    def update(self, classification, instanceClass):
        """ Update the performance objects according to a single instance. """
        if classification == -1:
            self.iDontKnow += 1
        else:
            self.iConfusionMatrix[classification][instanceClass] += 1
        self.iTotal += 1

        
    def reset(self):
        """ Resets the performance objects for a new evaluation cycle. """
        for i in range(self.numClass):
            for j in range(self.numClass):
                self.iConfusionMatrix[i][j] = 0
        self.iDontKnow = 0     
        self.iTotal = 0
