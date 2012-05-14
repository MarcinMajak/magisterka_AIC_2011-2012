#-------------------------------------------------------------------------------
# Name:        GAssist_Windowing.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: GAssist windowing scheme - creates data strata, manages data strata, sets of instances for learning from training data
#              Partitioning is achieved via initialization of this class, and strata iteration is handled within this class, so must be called externally, instance sets sent from this class.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Constants import * 
from random import *

class Windowing:
    def __init__(self, e, maxIterations):
        self.env = e
        self.trainSet = self.env.getTrainSet()
        self.numStrata = cons.numStrata  #Have this be a command line parameter where 1 is standard learning using all data.   will it work as is with 1?
        self.numClasses = self.env.getNrActions()
        self.strata = []  
        self.currentIteration = 0
        self.lastIteration = False
        self.maxIterations = maxIterations
        
        #under construction
        self.totalEval = True
        print "Creating Windows."
        self.createStrata() #Construct the strata
    
    
    def createStrata(self):
        """  Takes the Training dataset and randomly and in a balanced manner separates it into a user defined number of strata for learning."""
        # Construct empty nested lists
        tempStrata = [] #len(numStrata)
        instancesOfClass = [] #len(numClasses)
        for i in range(self.numStrata):
            tempStrata.append([])
            self.strata.append([])
        for i in range(self.numClasses):
            instancesOfClass.append([])

        #Strata Construction
        numInstances = len(self.trainSet)
        for i in range(numInstances): #separates instances into lists by class
            instancesOfClass[int(self.trainSet[i][1])].append(self.trainSet[i])  
        
        for i in range(self.numClasses): #class-wise addition to strata to keep strata class-balanced
            stratum = 0
            count = len(instancesOfClass[i]) #number of instances of given class
            while count >= self.numStrata: # handles most of the distribution
                pos = randint(0, count-1)
                tempStrata[stratum].append(instancesOfClass[i][pos]) #adds representative of class 
                instancesOfClass[i].pop(pos)
                stratum = (stratum+1)%self.numStrata #alternates adding an instance to each strata
                count -= 1
            while count > 0: #catches the last few instances, where there may not be enough for each strata
                stratum = randint(0, self.numStrata-1)
                tempStrata[stratum].append(instancesOfClass[i][0]) #zero here grabs the last remaining instance (there can be only one for each strata at best)
                instancesOfClass[i].pop(0)
                count -= 1
                
        for i in range(self.numStrata):
            num = len(tempStrata[i])
            for j in range(num):
                self.strata[i].append(tempStrata[i][j])
                
                
    def setEvalScope(self, evalAll):
        """ To handle initial, periodic and final evaluations of agents on the entire data sample this method shifts from windowing to looking at the entire training sample.
            Note specified originally, defaults to True for an initial evaluation of all agents prior to learning. """
        if evalAll:
            self.totalEval = True #examining all training samples.
        else:
            self.totalEval = False #windowing


    def newIteration(self): #should only be called during learning iterations - not initial, stops or final. called every evolvepop.
        """ Updates the Strata handeling for the next iteration. """
        self.currentIteration += 1
        
        if self.numStrata > 1:
            return True
        #return False
        return True #quick (inefficient fix for window size 1 bug)  All agents are re-evaluated each iteration) - more detailed fix is to set (isEvaluated) to False whenever an agent is mutated, crossover'd or changed in any way.
    
    
    def numVersions(self):
        """ Returns the number of data partitions currently being utilized. Switches to one on the final iteration. """
        if self.totalEval:
            return 1
        return self.numStrata
    
    
    def getInstances(self):
        """ Returns a single strata of instances until last iteration when entire dataset is returned instead. """
        if self.totalEval:
            return self.trainSet
        else:
            return self.strata[self.currentIteration%self.numStrata]
        
        
    def getCurrentVersion(self):
        """ Returns the segment number of the particular partition currently in use. """
        if self.totalEval:
            return 0 #references the one big chunk.
        return self.currentIteration%self.numStrata #(0,1)