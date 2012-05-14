#-------------------------------------------------------------------------------
# Name:        GAssist_Performance.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Handles performance updates 
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Constants import * 
import math

class GAssistPerformance:
    def __init__(self, nClass): 
        self.numAliveRules = 0 #number of rules that have matched at least one instance.
        self.totalInstances = 0 #tracks all instances examined
        self.okInstances = 0 #correct prediction counter
        
        self.accuracy = 0.0
        self.fitness = 0.0

        self.utilRules = [] #tracks whether the rule was used. ****
        self.accurateRule = []
        
        self.positionRuleMatch = 0  #new
        self.isEvaluated = False
        self.numClass = nClass  #Number of classes
   

    def resetPerformance(self, numRules):
        """ Reset the performance parameters. """
        self.numAliveRules = 0
        self.totalInstances = 0 #tracks all instances
        self.okInstances = 0 #correct prediction counter
        self.accuracy = 0.0
        self.fitness = 0.0  
        self.utilRules = []
        self.accurateRule = []
        
        for i in range(numRules):
            self.utilRules.append(0)
            self.accurateRule.append(0)

        self.isEvaluated = False  #check that this maintains rules not modified. where is resetPerformance called?/????????????????????????????
        
        
    def addPrediction(self, predicted, real):
        """ Function used to inform PerformanceAgent of each example classified during the training stage. """
        self.totalInstances += 1
        if(int(predicted) != -1):
            if self.utilRules[self.positionRuleMatch] == 0 and self.positionRuleMatch < len(self.utilRules): # if it's not the default rule position
                self.numAliveRules += 1
            self.utilRules[self.positionRuleMatch] += 1  #tracks whether the rule was used.
        if int(predicted) == int(real):
            self.okInstances += 1
            self.accurateRule[self.positionRuleMatch] += 1

    #def calculatePerformance(self, mdlfit): 
    def calculatePerformance(self, globalMDL, ind): 
        """ Calculates the accuracy and fitness parameters. """
        self.accuracy = self.okInstances / float(self.totalInstances)
        mdlFit = 0.0
        if cons.useMDL:
            mdlFit = globalMDL.mdlFitness(ind)
            
        penalty = 1.0
        if self.numAliveRules < cons.sizePenaltyMinRules:  #Agents have fitness penalty if the number of rules it has is below the threshold.
            #print "Agent with " + str(self.numAliveRules) + " rules, has been penalized. "
            penalty = (1 - 0.025 * (cons.sizePenaltyMinRules-self.numAliveRules))
            if penalty <= 0:
                penalty = 0.01
            penalty *= penalty
        if cons.useMDL:
            self.fitness = mdlFit / float(penalty) #accuracy not a part of fitness???
        else:
            self.fitness = self.accuracy
            self.fitness *= self.fitness
            self.fitness *= penalty
        self.isEvaluated = True
        
        
    def getAccuracy(self):
        """ Returns accuracy """
        return self.accuracy
        

    def getActivationsOfRule(self, rule):
        """ Get number of times a given rule has been activated. """
        return self.utilRules[rule]


    def controlBloatRuleDeletion(self):
        """ Determine which rules are to be deleted. """
        nRules = len(self.utilRules) # number of rules in agent
        minRules = cons.ruleDeletionMinRules  #12
        rulesToDelete = []
        countDeleted = 0
        if nRules > minRules: # delete rules only if there are more than the minimum size for an agent.
            for i in range(nRules):
                if self.utilRules[i] == 0:
                    rulesToDelete.append(i)
                    countDeleted += 1
                    #15 - 8 < 10
            if (nRules - countDeleted) < minRules: # if deletion has dropped the nRules below the min
                    #10 - (15-8) = 3 
                rulesToKeep = minRules - (nRules - countDeleted)  #how many of the 'to be deleted rules' should be salvaged to maintain the min.
                for i in range(rulesToKeep):
                    pos = randint(0,countDeleted-1) #pick a random rule to save.
                    rulesToDelete.pop(pos)
                    countDeleted -= 1
        #print "Deleting "+ str(len(rulesToDelete)) + " rules from agent."
        return rulesToDelete  #problem - rules to delete is too big sometimes, fix above


    def getFitness(self):
        """ Returns fitness """
        return self.fitness
    

    def getNumAliveRules(self):
        """ Returns the number of Alive rules """
        return self.numAliveRules 


    def getIsEvaluated(self):
        """ Returns whether this rule has already been evaluated. """
        return self.isEvaluated


    def setIsEvaluated(self, eval):
        """ Set whether this rule has now been evaluated. """
        self.isEvaluated = eval