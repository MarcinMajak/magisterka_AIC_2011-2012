#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Import modules
from GALE_Board import *
import time
#from pylab import * #Used for the graphing method

class GALE:
    def __init__(self, e, outFileString, popOutFileString, bitLength, CVparts, graphPerform, xSize, ySize, distribution):
        """ Initialize GALE algorithm objects:  """
        #Output Filenames
        self.outFile = outFileString    #Saves tracking values
        self.popFile = popOutFileString #Saves evaluation and print out of final populations (at pre-specified iteration stops)
       
        #Board Parameters
        self.env = e
        self.xSize = xSize
        self.ySize = ySize
        self.brdDist = distribution
       
        #Used do determine how assessments done.  is there testing data? if CVparts >1 than yes.
        self.CVpartitions = CVparts
        
        # Track learning time
        self.startLearningTime = 0
        self.endLearningTime = 0
        self.evalTime = 0
        
        #Other Objects
        self.attributeGenList = []
        self.performanceTrackList = [[0],[0],[0],[0],[0],[0],[0],[0]]
        self.graphPerform = graphPerform
        self.bitLength = bitLength
        self.trainAccuracy  = 0
        self.testAccuracy  = 0        
        
        
    #METHODS SETTING PARAMETERS *****************************************************************************
    def setNumberOfTrials(self, trials, iterList):
        """ Resets the maximal number of learning iterations in an LCS run."""
        self.maxProblems = trials
        self.iterStops = iterList        
        
        
    def setTrackingIterations(self, trkCyc):
        """ Resets the parameters for how often progress tracking is reset and sent to the output file."""
        self.trackIter = trkCyc

        
    #KEY METHOD *********************************************************************************************       
    def runGALE(self):
        """ Method to run the GALE algorithm. """
        #***************************************************************
        try:
            pW = open(self.outFile+'.txt','w')     
        except IOError, (errno, strerror):
            print ("I/O error(%s): %s" % (errno, strerror))
            raise
        #***************************************************************
        
        self.startLearningTime = time.time()    #TIMER STARTS
        
        # Initialize Board/Population of Agents
        board = GALEBoard(self.xSize, self.ySize, self.brdDist, self.env.getTrainSet(), self.env, self.env.getTestSet())
        
        pW.write("Iteration\tNumAgents\tBestAccuracy\tBestComplexity\tBestNoMatch\tBestGenerality\tAveAccuracy\tAveComplexity\tAveNoMatch\tAveGenerality\n")
        iteration = 0
        tempTimeA = 0
        tempTimeB = 0
        
        #Get and print initial tracking stats
        board.computeStatistics()
        trackInit = board.getTracking()
        self.writePerformance(pW, trackInit, -1)   
              
        print "Begin Evolution Cycles:"        
        while iteration < self.maxProblems:
            #Manages the evolution and tracking of the GALE algorithm
            board.evolveBoard() #Entire board is evolved once.
            track = board.getTracking()
            if (iteration+1)%self.trackIter == 0:
                self.writePerformance(pW, track, iteration) 
            
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
                
                board.computeStatistics()   #Update statistics to be current before printing output
                trainEval = []
                testEval = []
                if self.CVpartitions > 1: 
                    trainEval = track   #Evaluation done already  
                    #testPerform = board.testPerform()  #check for accuracy HERE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    testAgent = board.testPerform() 
                    testPerform = testAgent.getPerformance()
                    accuracy = testPerform.getAccuracy()
                    noMatch = testPerform.getNoMatch()
                    testEval = [accuracy, noMatch]
                else:
                    trainEval = track
                    testEval = [None, None]
                    
                #Time management
                self.endLearningTime = time.time()
                tempTimeB = time.time() 
                self.evalTime += tempTimeB - tempTimeA  
                learnSec = self.endLearningTime - self.startLearningTime - self.evalTime
                
                #Report on current agent population
                bestAgent = board.getBestAgent()
                self.printCSEvaluation(popW, trainEval, testEval, learnSec, self.env, self.CVpartitions, bestAgent, iteration, testAgent) #track test eval from here!!!!
                
                #***************************************************************
                try:
                    popW.close()  
                except IOError, (errno, strerror):
                    print ("I/O error(%s): %s" % (errno, strerror))
                    raise     
                #***************************************************************

            iteration += 1
        
        #***************************************************************
        print "LCS Training and Evaluation Complete!"
        try:
            pW.close()
        except IOError, (errno, strerror):
            print ("I/O error(%s): %s" % (errno, strerror))
            raise  
        #***************************************************************
        

    def printCSEvaluation(self, popW, track, testEval, learnSec, env, CVPartition, bestAgent, iteration, testAgent):
        """ Makes an output file containing a complete evaluation of the current GALE agent population, and a print out of the current best agent (and all it's rules). """
        vector = bestAgent.getVector()
        
        # Two sections to the output: Learning Characteristics and Population/Agent Characteristics
        popW.write("Training:\n")
        popW.write("Iteration\tNumAgents\tBestAccuracy\tBestComplexity\tBestNoMatch\tBestGenerality\tAveAccuracy\tAveComplexity\tAveNoMatch\tAveGenerality\tRunTime(min)\n")
        popW.write(str(iteration+1) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) +"\t" + str(track[7]) + "\t" + str(track[8])+ "\t"  + str(learnSec/60) + "\n")
        popW.write("Testing Accuracy: \t TestNoMatch: \n")
        if CVPartition > 1:  #Print testing stats if CV has been utilized
            popW.write(str(testEval[0]) + "\t" + str(testEval[1])+"\n")
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
        popW.write("Condition\tAction\tPrune\tMatch\tAccuracy\tGenerality\tTestPrune\tTestMatch\n")
        
        # Prints out the rules of the best agent.
        for i in range(len(vector)):
            genCount = 0
            for j in range(len(vector[i])):
                if j == len(vector[i])-1:
                    popW.write("\t")
                    popW.write(str(vector[i][j]))
                else:
                    popW.write(str(vector[i][j]))
                    if vector[i][j] == cons.dontCare:
                        genCount += 1 
            tempAcc = 0
            if bestAgent.getMatch()[i] == 0:
                tempAcc = 0
            else:
                tempAcc = bestAgent.getPrune()[i]/float(bestAgent.getMatch()[i])      
            popW.write("\t" + str(bestAgent.getPrune()[i]) + "\t" +  str(bestAgent.getMatch()[i]) + "\t" + str(tempAcc) + "\t"  + str(genCount/float(len(vector[i])-1))+"\t" + str(testAgent.getPrune()[i]) + "\t" +  str(testAgent.getMatch()[i])+ "\n")

            
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
                
 
    def writePerformance(self, pW, track, iteration):
        """ Writes the tracking data.  """
        pW.write(str(iteration+1) + "\t" + str(track[0]) + "\t" + str(track[1]) + "\t" + str(track[2]) + "\t" + str(track[3]) + "\t" + str(track[4]) + "\t" + str(track[5]) + "\t" + str(track[6]) +"\t" + str(track[7]) + "\t" + str(track[8]) + "\n")
        print ("Iter: " + str(iteration+1) + "\t NumAgents: " + str(track[0]) + "\t BestAccuracy: " + str(track[1]) + "\t BestComplexity: " + str(track[2]) + "\t BestNoMatch: " + str(track[3]) +"\t BestGenerality: " + str(track[4]) + "\t AveAccuracy: " + str(track[5]) + "\t AveComplexity: " + str(track[6]) + "\tAveNoMatch: " + str(track[7]) + "\t AveGenerality: " + str(track[8]))

        # GRAPHING ******************************************************************
        if self.graphPerform: #NEEDS WORK - does not work at present!!!!!!!!!!!!
            self.performanceTrackList[0].append(iteration+1)
            self.performanceTrackList[1].append(track[0]/float(self.maxPopSize))
            self.performanceTrackList[2].append(track[1])
            self.performanceTrackList[3].append(track[2])#Needs to be divided!!!
            self.performanceTrackList[4].append(track[3])  
            self.performanceTrackList[5].append(track[4])          
            self.performanceTrackList[6].append(track[5])#Needs to be divided!!!        
            self.performanceTrackList[7].append(track[6]) 
            self.graphPerformance()
        # ***************************************************************************

    def graphPerformance(self):
        """ Uses pylab to graph tracking values while the algorithm is running for a real time perspective on algorithm performance. 
        CURRENTLY NOT FUNCTIONAL: """
        ion()   # Turn on interactive mode for interactive graphing
        line1 = plot(self.performanceTrackList[0], self.performanceTrackList[1],'g-')
        line2 = plot(self.performanceTrackList[0], self.performanceTrackList[2],'r-')
        line3 = plot(self.performanceTrackList[0], self.performanceTrackList[3],'b-')
        line4 = plot(self.performanceTrackList[0], self.performanceTrackList[4],'c-')
        line5 = plot(self.performanceTrackList[0], self.performanceTrackList[5],'y-')
        line6 = plot(self.performanceTrackList[0], self.performanceTrackList[6],'k-')
        line7 = plot(self.performanceTrackList[0], self.performanceTrackList[6],'m-')
        #axis([0, self.maxProblems, 0, 1])
        legend((line1,line2,line3,line4,line5,line6,line7),('NumAgents','BestAccuracy','BestComplexity','BestGenerality','AveAccuracy', 'AveComplexity', 'AveGenerality'),loc=4)
        xlabel('Training Iterations')
        title('LCS Performance Tracking')
        grid(True)
