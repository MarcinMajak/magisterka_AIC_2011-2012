#-------------------------------------------------------------------------------
# Name:        GAssist.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: GAssist algorithm - Initializes the algorithm objects, manages time tracking, writing progress - replaces control class of java version.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Pop import *
from GAssist_Windowing import *    #might not be needed here !!!!!!!check
from GAssist_Constants import *  
import time
#from pylab import *

class GAssist:
    def __init__(self, e, outFileString, popOutFileString, bitLength, CVparts, graphPerform):
        """ Initialize GAssist algorithm objects:  """
        #Output Filenames
        self.outFile = outFileString    #Saves tracking values
        self.popFile = popOutFileString #Saves evaluation and print out of final populations (at pre-specified iteration stops)
       
        #Board Parameters
        self.env = e
       
        #Used do determine how assessments done.  is there testing data? if CVparts >1 than yes.
        self.CVpartitions = CVparts
        
        # Track learning time
        self.startLearningTime = 0
        self.endLearningTime = 0
        self.evalTime = 0
        
        #Other Objects
        self.attributeGenList = []
        self.performanceTrackList = [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
        self.graphPerform = graphPerform
        self.bitLength = bitLength
        self.trainAccuracy  = 0
        self.testAccuracy  = 0     
        
        #Initialization Parameters
        self.smartInit = False
        self.cwInit = False
        
        
    #METHODS SETTING PARAMETERS *****************************************************************************
    def setNumberOfTrials(self, trials, iterList):
        """ Resets the maximal number of learning iterations in an LCS run."""
        self.maxProblems = trials
        self.iterStops = iterList        
        
        
    def setTrackingIterations(self, trkCyc):
        """ Resets the parameters for how often progress tracking is reset and sent to the output file."""
        self.trackIter = trkCyc


    def setInitialization(self, initMethod):
        """ Sets the rule initialization method.  Can be 'none' which is a random initialization, 'smart' which is similar to standard covering in XCS
        and 'cw' which is similar to smart, but with equal class opportunity. """
        
        if initMethod == "smart":
            self.smartInit = True
            self.cwInit = False
            print "Smart initialization used!"
        elif initMethod == "cw":
            self.smartInit = True
            self.cwInit = True
            print "Smart and balanced initialization used!"
        else:
            self.smartInit = False
            self.cwInit = False   
            print "Default random initialization used!"
        

    #KEY METHOD *********************************************************************************************       
    def runGAssist(self):
        """ Method to run the GAssist algorithm. """
        #***************************************************************
        try:
            pW = open(self.outFile+'.txt','w')     
        except IOError, (errno, strerror):
            print ("I/O error(%s): %s" % (errno, strerror))
            raise
        #***************************************************************
        
        #Initialize Windowing - takes all data set instances as argument 
        windows = Windowing(self.env, self.maxProblems) 

        #Initialize the population
        population = GAssistPop(self.env, self.env.getTrainSet(), self.env.getTestSet(), windows, self.smartInit, self.cwInit, self.maxProblems)
                
        #Get and print initial tracking stats
        pW.write("Iteration\tWindow\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\n")
        trackInit = population.getTracking() # an initial evaluation should be done using the entire set. also - send window the iteration, instead of separate internal counter
        currentWindow = windows.getCurrentVersion()
        self.writePerformance(pW, trackInit, -1, currentWindow) #Again this may need to be removed here - initial population performance writing.******** START HERE START HERE START HERE START HERE
          
        #TIMER STARTS
        self.startLearningTime = time.time()     
        tempTimeA = 0
        tempTimeB = 0
        
        iteration = 0         
        print "Begin Evolution Cycles:"     
        while iteration < self.maxProblems:
            #print "Starting iteration: " + str(iteration) #Debugging
            #Run a single evolution iteration
            lastIteration = (iteration == self.maxProblems-1);
            
            cons.setLearning(iteration/self.maxProblems)
            windows.setEvalScope(False) #evaluations will take place within a window.
            population.evolvePopulation(lastIteration, iteration)
            #Track the population statistics
            track = population.getTracking() #return [self.bestFitness, self.bestAccuracy, self.bestNumRules, bestAliverules, self.bestGenerality, self.averageFitness, self.averageAccuracy, self.averageNumRules, self.averageNumRulesUtils, self.averageGenerality] # 10 items
            if (iteration+1)%self.trackIter == 0:
                self.writePerformance(pW, track, iteration, windows.getCurrentVersion()) 
                #bestAgent = population.getBest()
                #self.printCurrentBest(bestAgent)
                
            #Evaluate current status at pre-specified stop iterations. *************************************************************************
            if iteration + 1 in self.iterStops:
                tempTimeA = time.time()
                #***************************************************************
                try:  
                    popW = open(self.popFile + '.'+ str(iteration + 1)+'.txt','w')   #New for pop output - now have to open one for each iteration setting
                except IOError, (errno, strerror):
                    print ("I/O error(%s): %s" % (errno, strerror))
                    raise
                #***************************************************************

                windows.setEvalScope(True) #evaluations will take place within the entire training set
                
                population.trainPopulation()
                track = population.getTracking()
                
                bestAgent = population.getBest()
                bA = bestAgent.clone()

                trainEval = track
                testEval = []
                if self.CVpartitions > 1: 
                    #Evaluate best agent using the test set.
                    #inst = windows.getInstances()
                    inst = self.env.getTestSet()
                    testAccuracy = bA.testEvaluateAgent(inst, population.globalMDL)
                    testEval = [testAccuracy]
                    
                else:
                    testEval = [None]
                 
                #Time management
                self.endLearningTime = time.time()
                tempTimeB = time.time() 
                self.evalTime += tempTimeB - tempTimeA  
                learnSec = self.endLearningTime - self.startLearningTime - self.evalTime
                
                self.printCSEvaluation(popW, trainEval, testEval, learnSec, self.env, self.CVpartitions, bestAgent, iteration)
                
                #***************************************************************
                try:
                    popW.close()  
                except IOError, (errno, strerror):
                    print ("I/O error(%s): %s" % (errno, strerror))
                    raise     
                #***************************************************************
                population.returnPopulation()
            iteration += 1
        
        #***************************************************************
        print "LCS Training and Evaluation Complete!"
        try:
            pW.close()
        except IOError, (errno, strerror):
            print ("I/O error(%s): %s" % (errno, strerror))
            raise  
        #***************************************************************
        

    def printCSEvaluation(self, popW, track, testEval, learnSec, env, CVPartition, bestAgent, iteration):
        """ Makes an output file containing a complete evaluation of the current GALE agent population, and a print out of the current best agent (and all it's rules). """
        vector = bestAgent.getVector()
        print "Stop Point Agent Evaluation::"
        # Two sections to the output: Learning Characteristics and Population/Agent Characteristics
        popW.write("Training:\n")
        popW.write("Iteration\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\tRunTime(min)\n")
        popW.write(str(iteration+1) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) +"\t" + str(track[7]) + "\t" + str(track[8]) + "\t" + str(track[9])+ "\t" + str(learnSec/60) + "\n")
        print ("Eval. Point: " + str(iteration+1) + "\t Window: All" + "\t BestFitness: " + str(track[0]) + "\t BestAccuracy: " + str(track[1]) + "\t BestNumRules: " + str(track[2]) + "\t BestAliveRules: " + str(track[3]) +"\t BestGenerality: " + str(track[4]) + "\t AveFitness: " + str(track[5]) + "\t AveAccuracy: " + str(track[6]) + "\tAveNumRules: " + str(track[7]) + "\t AveNumRulesUtils: " + str(track[8])+ "\t AveGenerality: " + str(track[9]))
        popW.write("Testing Accuracy:\n")
        if CVPartition > 1:  #Print testing stats if CV has been utilized
            popW.write(str(testEval[0])+"\n")
            print "Testing Accuracy: " +str(testEval[0])
        else:
            popW.write("NA\n")
        popW.write("Population Characterization:\n")  
        popW.write("WildSum\n")
        
        # Print the attribute labels
        headList = env.getHeaderList()
        for i in range(len(headList)-1):    # Added the -1 to get rid of the Class Header
            if i < len(headList)-2:
                popW.write(str(headList[i])+"\t")
            else:
                popW.write(str(headList[i])+"\n")
                
        wildCount = self.characterizePop(vector)
        self.attributeGenList = self.condenseToAttributes(wildCount,self.bitLength) #List of the bitLength corrected wild counts for each attribute.
                
        # Prints out the generality count for each attribute.
        for i in range(len(self.attributeGenList)):
            if i < len(self.attributeGenList)-1:
                popW.write(str(self.attributeGenList[i])+"\t")
            else:
                popW.write(str(self.attributeGenList[i])+"\n")        
        
        popW.write("Ruleset Population: \n")
        popW.write("Condition\tAction\tCorrect\tMatch\tAccuracy\tGenerality\n")
        
        # Prints out the rules of the best agent.
        for i in range(len(vector)):
            genCount = 0
            for j in range(len(vector[i])):
                if j == len(vector[i])-1: #Write Action/Class
                    popW.write("\t")
                    popW.write(str(vector[i][j]))
                else: #Write Condition
                    popW.write(str(vector[i][j]))
                    if vector[i][j] == cons.dontCare:
                        genCount += 1 
            tempAcc = 0
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i]/float(bestAgent.agnPer.utilRules[i])      
            popW.write("\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" +  str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc)+"\t" + str(genCount/float(len(vector[i])-1))+ "\n")
            
        if cons.defaultClass != "disabled":
            #Print the default rule.
            for i in range(len(vector[0])-1):
                popW.write(cons.dontCare)
            popW.write("\t"+str(bestAgent.defaultClass))
            
            tempAcc = 0
            i = len(vector)
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i]/float(bestAgent.agnPer.utilRules[i])      
            popW.write("\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" +  str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc)+"\t" + str(1)+ "\n")
            

    def characterizePop(self,vector):
        """ Make a list counting the #'s in each attribute position across the best agent's rule set. """
        countList = []
        for x in range(len(vector[0])-1):
            countList.append(int(0))
        for i in range(len(vector)):
            for j in range(len(vector[i])-1):
                if vector[i][j] == cons.dontCare:
                    countList[j] += 1
        return countList
    
    
    def condenseToAttributes(self, tempList, bitLength):
        """ Takes the results of 'characterizePop' and condenses the count values down to include all bits coding a single attribute """
        temp = 0
        newList = []
        for i in range(len(tempList)):
            if (i+1)%int(bitLength) != 0:  #first run it will be 1
                temp = temp + tempList[i]
            else:
                newList.append(temp + tempList[i])
                temp = 0
        return newList
                
 
    def writePerformance(self, pW, track, iteration, currentWindow):
        """ Writes the tracking data.  """
        pW.write(str(iteration+1) + "\t" +str(currentWindow) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) +"\t" + str(track[7]) + "\t" + str(track[8]) + "\t" + str(track[9])+ "\n")
        print ("Iter: " + str(iteration+1) + "\t Window: " + str(currentWindow) + "\t BestFitness: " + str(track[0]) + "\t BestAccuracy: " + str(track[1]) + "\t BestNumRules: " + str(track[2]) + "\t BestAliveRules: " + str(track[3]) +"\t BestGenerality: " + str(track[4]) + "\t AveFitness: " + str(track[5]) + "\t AveAccuracy: " + str(track[6]) + "\tAveNumRules: " + str(track[7]) + "\t AveNumRulesUtils: " + str(track[8])+ "\t AveGenerality: " + str(track[9]))
        #"Iteration\tBestFitness\tBestAccuracy\tBestNumRules\tBestAliverules\tBestGenerality\tAverageFitness\tAverageAccuracy\tAverageNumRules\tAverageNumRulesUtils\tAverageGenerality\n")

    # DEBUGGING
    def printCurrentBest(self, bestAgent):
        """ Used for debugging - prints out the best agent of the current iteration. """
        vector = bestAgent.getVector()
        
        print "Ruleset Population:"
        print "Condition\tAction\tCorrect\tMatch\tAccuracy"
        
        # Prints out the rules of the best agent.
        for i in range(len(vector)):
            print len(vector)
            print len(bestAgent.agnPer.utilRules)
            tempAcc = 0
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i]/float(bestAgent.agnPer.utilRules[i])   
                
            print str(vector[i])+ "\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" +  str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc)
            
        #Print the default rule.    
        if cons.defaultClass != "disabled":
            tempAcc = 0
            i = len(vector)
            if bestAgent.agnPer.utilRules[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.agnPer.accurateRule[i]/float(bestAgent.agnPer.utilRules[i]) 
                
            print "Default Rule #### - " +str(bestAgent.defaultClass) + "\t" + str(bestAgent.agnPer.accurateRule[i]) + "\t" +  str(bestAgent.agnPer.utilRules[i]) + "\t" + str(tempAcc)+"\t" + str(1)
        