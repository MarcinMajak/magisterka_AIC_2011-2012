#-------------------------------------------------------------------------------
# Name:        GAssist_Pop.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Initializes population and handles all population level functions.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Constants import *  
from GAssist_Agent import *
from GAssist_Sampling import *
from GAssist_Global_MDL import *
from random import *
from copy import deepcopy
import copy
import math

class GAssistPop:    
    def __init__(self, e, inSet, testSet, w, smart, cw, maxProblems): #self.env, self.env.getTrainSet(), self.env.getTestSet(), windows, self.smartInit, self.cwInit
        print "Initializing Population"
        self.env = e
        self.windows = w
        self.maxProblems = maxProblems
        self.tempPop = []
        self.pop = []  #The population of Pitts individuals

        #Data/ Instance Variables
        self.instances = inSet
        self.testSet = testSet
        
        #Windowing/Version Handling
        self.numVersions = cons.numStrata
        self.bestAgents = [] # A list of agent/individual objects which represent the best found for each version/window.
        for i in range(0,self.numVersions): #Makes an empty list with enough slots to hold all best agents representing each version/window.
            self.bestAgents.append(None)
            
        # Initialize Population of Agents
        print "Making " + str(cons.popSize) + " agents."
        for i in range(cons.popSize):     #Initialize the specified population Size of agents/individuals
            self.pop.append(None)

        for i in range(cons.popSize):
            self.pop[i] = GAssistAgent(self.env, smart, cw)

        print "Population Initialization Complete."
        
        #Tracking Statistics
        self.trackBestFitness = 0.0
        self.bestFitness = 0.0
        self.bestAccuracy = 0.0
        self.bestNumRules = 0.0
        self.bestAliverules = 0.0
        self.bestGenerality = 0.0
        
        self.averageFitness = 0.0
        self.averageAccuracy = 0.0
        self.averageNumRules = 0.0
        self.averageNumRulesUtils = 0.0
        self.averageGenerality = 0.0
        
        self.last10Accuracy = [0.0 for i in range(10)]
        self.last10IterationsAccuracyAverage = 0.0
        self.iterationsSinceBest = 0
        self.countStatistics = 0
        self.firstIteration = True
        
        self.iterationNichingDisabled = None  # add this to final output

        self.globalMDL = GAssistGlobalMDL(self.env) # initializes this class for MDL calculations.

        print "Calculating initial performance." 
        self.doEvaluation(self.pop) #done using whole training set.
        self.calculateStatistics()



    def checkBestIndividual(self):
        """ Identifies and sets the best agent for this learning iteration. - Responsible for updating bestIndiv[]. """
        best = self.getBest()
        currVer = self.windows.getCurrentVersion()
        if self.bestAgents[currVer] == None:
            self.bestAgents[currVer] = best.clone()
        else:
            if best.compareToIndividual(self.bestAgents[currVer]):
                self.bestAgents[currVer] = best.clone()
        
        
    def getPopClone(self):
        """ Returns a population clone so that evaluations can be called from GAssist module non destructively. """
        objClone = deepcopy(self)
        return objClone
    
           
    def doEvaluation(self, pop):
        """ Evaluates the performance of each agent in the population."""
        for i in range(cons.popSize):   #For each agent in the population
            if pop[i].agnPer.isEvaluated == False:
                self.evaluateClassifier(pop[i])

    
    def evaluateClassifier(self, ind):
        """ Evaluates the performance of each agent in the population."""
        instanceWindow = self.windows.getInstances()
        ind.agnPer.resetPerformance(ind.getNumRules())
        for j in range(len(instanceWindow)): #for each instance in the window
            real = instanceWindow[j][1] 
            predicted = ind.classify(instanceWindow[j])  
            ind.agnPer.addPrediction(predicted,real)
            
        ind.agnPer.calculatePerformance(self.globalMDL, ind)
        
        if cons.doRuleDeletion:
            ind.deleteRules(ind.agnPer.controlBloatRuleDeletion())
            
            
    def calculateStatistics(self):  #ERROR HERE INFINATE LOOP
        """ Calculates the statistics of the current agent population """
        popLength = cons.popSize
        sumFitness = 0.0
        sumAccuracy = 0.0
        sumNumRules = 0.0
        sumNumRulesUtils = 0.0
        sumGenerality = 0.0
        
        for i in range(popLength):
            #self.pop[i].defaultClassCheck()  # DEBUGGING
            agent = self.pop[i]
            sumFitness += agent.agnPer.getFitness()
            sumAccuracy += agent.agnPer.getAccuracy()
            sumNumRules += agent.getRealNumRules()
            sumNumRulesUtils += agent.agnPer.getNumAliveRules()
            sumGenerality +=  agent.getGenerality()

        self.averageFitness = sumFitness / popLength
        self.averageAccuracy = sumAccuracy / popLength
        self.averageNumRules = sumNumRules / popLength
        self.averageNumRulesUtils = sumNumRulesUtils / popLength
        self.averageGenerality = sumGenerality / popLength

        #Get the Best info
        bestAgent = self.getBest()  #finish this!! figure out how the best windowing works.

        self.bestFitness = bestAgent.agnPer.getFitness()
        self.bestAccuracy = bestAgent.agnPer.getAccuracy()
        self.bestNumRules = bestAgent.getRealNumRules()
        self.bestAliverules = bestAgent.agnPer.getNumAliveRules()
        self.bestGenerality = bestAgent.getGenerality()

        self.last10Accuracy.pop(0)
        self.last10Accuracy.append(self.bestAccuracy)

        self.bestOfIteration(bestAgent.agnPer.getFitness())  #how to present fitness. there are more than one. (old and new)
        self.countStatistics += 1


    def bestOfIteration(self, itBestFit):
        """ Determines the best agent for the current iteration. """
        if self.iterationsSinceBest == 0: #first iteration only
            self.iterationsSinceBest += 1
            if self.firstIteration:
                self.trackBestFitness = itBestFit
                self.firstIteration = False
                print "Set initial best fitness."
        else: 
            newBest = False
            if cons.useMDL:
                if itBestFit < self.trackBestFitness: #when using MDL it appears that smaller fitness values are more fit. CHECK THIS!!!!!!!!!!
                    newBest = True #since self.bestFitness starts at 0, nothing can be lower.
            else:
                if itBestFit > self.trackBestFitness:  #otherwise higher fitness values are better - indicative of higher accuracy.
                    newBest = True
            if newBest:
                print "NEW BEST AGENT FOUND"
                self.trackBestFitness = itBestFit #this is where self.bestFitness needs to be set.
                self.iterationsSinceBest = 1
            else:
                self.iterationsSinceBest += 1
        
        i = self.countStatistics - 9 
        if i < 0:  #if this is one of the first 10 cycles
            i = 0
        max = self.countStatistics + 1 
        num = max - i  #num handles average calculations when there are not yet 10 iterations
        self.last10IterationsAccuracyAverage = 0  
        
        if i < 1: #during the first 10 iterations
            for j in range(i,max): #set up for tracking all values, not just the last 10
                self.last10IterationsAccuracyAverage += self.last10Accuracy[j]
        else:
            for j in range(10):
                self.last10IterationsAccuracyAverage += self.last10Accuracy[j]
                
        self.last10IterationsAccuracyAverage /= float(num)
        
                
    def getBest(self):
        """  Returns the best agent in the population. """
        posWinner = 0
        for i in range(len(self.pop)):
            if self.pop[i].compareToIndividual(self.pop[posWinner]):  
                posWinner = i
        return self.pop[posWinner]


    def getWorst(self, offspring):
        """ Returns the location of the worst agent in the population. """
        posWorst = 0
        for i in range(len(offspring)):
            if offspring[i].compareToIndividual(offspring[posWorst]) == False:
                posWorst = i
        return posWorst
    

    def resetBestStats(self):
        """ Reinitializes the iterations since best. """
        self.iterationsSinceBest = 0

        
    def evolvePopulation(self, lastIteration, iteration):
        """ Directs the evolution of the agent population.  Calls the major operations. """
        res1 = self.windows.newIteration() # Shifts data set portion to new window. Everything done during this evolution is from the perspective of this window (including best agent)
        res2 = self.runTimers(iteration) 
        
        if res1 or res2:
            self.setModified() #isEvaluated always reset for all agents when window switches
            
        #GA Cycle - Main learning operations
        self.pop = self.doTournamentSelection() #TournamentSelects the population to make a new population of best agents.
        if cons.defaultClass == "disabled":
            offspring = self.doCrossover()
        else:
            offspring = self.doNicheCrossover() # new population object
        self.doMutation(offspring) 
        self.doEvaluation(offspring) 
        self.pop = self.replacementPolicy(offspring, lastIteration) 
        self.checkBestIndividual()
        self.calculateStatistics()
        self.checkNichingStatus(iteration)
        
        
    def trainPopulation(self):
        """ Handles the full training evaluation - not repetative because final replacement policy is implemented here. """
        self.tempPop = deepcopy(self.pop)
        self.setModified()
        self.pop = self.replacementPolicy(self.pop, True)
        self.doEvaluation(self.pop)
        self.calculateStatistics()
 
 
    def returnPopulation(self):
        """ Reverts to original population from before train/test evaluation. """
        self.pop = self.tempPop
        
        
    def runTimers(self, iteration):
        """ Perform the timer functions """
        res1 = self.globalMDL.newIteration(iteration, self)
        res2 = self.timerBloat(iteration)
        
        if res1 or res2:
            return True
        return False


    def timerBloat(self, iteration):
        """ Resets some new bloat constants at timed checkpoints. """
        cons.setRuleDelete(iteration >= cons.iterationRuleDeletion)
        cons.setHierarchy(iteration >= cons.iterationHierarchicalSelection)
        
        if iteration == self.maxProblems - 1:
            cons.setDeleteMinRules(1)
            return True
        
        if iteration == cons.iterationRuleDeletion:
            return True
        
        return False


    def doTournamentSelection(self):
        """ Does Tournament Selection without replacement """
        selectedPopulation = [None for i in range(cons.popSize)]
        numNiches = 0
        if cons.nichingEnabled: #Initially niching is used but once niche accuracies stabilize, it shuts off.
            numNiches = self.env.getNrActions()
        else:   #Niche based tournament selections stops.
            numNiches = 1
        
        pools = [[] for i in range(numNiches)] # will hold lists of agent ID's 
        nicheCounters = [0 for i in range(numNiches)]
        nicheQuota = int(cons.popSize/numNiches)
        for i in range(numNiches): #balanced number of times that selection occurs in each niche
            nicheCounters[i] = nicheQuota
            
        for i in range(cons.popSize): # For each agent in the population
            #There can be only one.
            niche = self.selectNicheWOR(nicheCounters) #Selects a niche from which to perform selection. - nicheCounters is reduced for some niche.
            winner = self.selectCandidateWOR(pools[niche], niche)  #Selects a single agent ID from given niche
            for j in range(cons.tournamentSize): # Do tournament selection - basically the size becomes cons.tournamentSize + 1 - as its programmed.
                candidate = self.selectCandidateWOR(pools[niche], niche) # pick another agent ID from given niche
                if self.pop[candidate].compareToIndividual(self.pop[winner]):
                    winner = candidate
            selectedPopulation[i] = self.pop[winner].clone()  #implement copy() - base on clone. 
                              
        return selectedPopulation #same array format as self.pop


    def selectNicheWOR(self, quotas):
        """ Selects a niche without replacement. """
        num = len(quotas) #number of niches
        if num == 1: #if niching is turned off
            return 0 
        
        total = 0
        for i in range(num): 
            total += quotas[i] #should be = or a little less than popSize 
        if total == 0:  #This is a catch for the case that the initial total is less than popSize.
            return randint(0, num-1) #Returns random niche ID.
        
        pos = randint(0, total - 1) #pick a random agent ID.
        total = 0  #seems odd at first but this re-do of total needs to be here.
        for i in range(num):
            total += quotas[i] #like roulete wheel selection for a niche.
            if pos < total:
                quotas[i] -= 1 
                return i # Returns niche ID
            
        print "Error - selectNicheWOR"
        return -1
        

    def selectCandidateWOR(self, pool, whichNiche):
        """ Selects a candidate without replacement. """
        if len(pool) == 0: #build the pool
            self.initPool(pool, whichNiche)
            if len(pool) == 0: # if pool is still empty - return a random agent from the whole pop.
                return randint(0, cons.popSize - 1)
        
        pos = randint(0, len(pool)-1) #pick an agent ID from this pool
        elem = int(pool[pos])
        pool.pop(pos) 
        return elem
        

    def initPool(self, pool, whichNiche):
        """ Builds the pool list for the specified niche.  Pool list is an array of agent ID references. """
        if cons.nichingEnabled:
            for i in range(cons.popSize):
                if self.pop[i].getNiche() == whichNiche: #if the agent is part of the selected niche
                    pool.append(i)
        else:
            for i in range(cons.popSize):
                pool.append(i)
        

    def checkNichingStatus(self, iteration):
        """ Checks whether niching is still needed for selection - Niching is turned off once the tournament with 
        niche preservation is used until the best individuals of each default class have similar training accuracy. """
        
        if cons.nichingEnabled:
            numNiches = self.env.getNrActions()
            counters = [0 for i in range(numNiches)]
            nicheFit = [0.0 for i in range(numNiches)]
            for i in range(len(self.pop)):
                niche = self.pop[i].getNiche()
                counters[niche] += 1
                indAcc = self.pop[i].agnPer.getAccuracy()
                if indAcc > nicheFit[niche]:
                    nicheFit[niche] = indAcc
            
            if len(cons.accDefaultRules[0]) == 15:  #If there have been at least 15 iterations.
                for i in range(numNiches):
                    cons.accDefaultRules[i].pop(0) # remove oldest accuracy
            
            for i in range(numNiches):
                cons.accDefaultRules[i].append(nicheFit[i])
 
            if len(cons.accDefaultRules[0]) == 15:
                aves = []
                for i in range(numNiches):
                    aveN = self.getAverage(cons.accDefaultRules[i])
                    aves.append(aveN)
        
                dev = self.getDeviation(aves)
                if dev < 0.005:
                    print "Iteration "+str(iteration)+", niching disabled"
                    cons.setNichingStatus(False)
                    self.iterationNichingDisabled = iteration
        
        
    def doNicheCrossover(self):
        """ Arranges the crossover within each niche and selects parents. Does not do actual crossing over of agents. """
        countCross = 0
        numNiches = self.env.getNrActions()
        parents = [[] for i in range(numNiches)] #separates potential parents(agents) into groups based on niche.
        parent1 = None
        parent2 = None
        offspring = [None,None]
        offspringPopulation = [[] for i in range(cons.popSize)]
        
        for i in range(cons.popSize): #construct parents
            niche = self.pop[i].getNiche()
            parents[niche].append(i)
            
        for i in range(numNiches): # do crossover separately for each niche
            size = len(parents[i]) # number of parents in this niche
            samp = Sampling(size)
            p1 = -1
            for j in range(size): #for each parent.
                if random() < cons.probCrossover:
                    if p1 == -1:  #gets a first parent and takes up a step when a crossover was successful
                        p1 = samp.getSample()
                    else:
                        p2 = samp.getSample()
                        pos1 = parents[i][p1]
                        pos2 = parents[i][p2]
                        parent1 = self.pop[pos1]
                        parent2 = self.pop[pos2]
                        
                        offspring = parent1.crossoverClassifiers(parent2)
                        offspringPopulation[countCross] = offspring[0]
                        countCross += 1
                        offspringPopulation[countCross] = offspring[1]
                        countCross += 1
                        p1 = -1
                else:
                    pos = parents[i][samp.getSample()]
                    offspringPopulation[countCross] = self.pop[pos].clone()
                    countCross += 1
            if p1 != -1:
                pos = parents[i][p1]
                offspringPopulation[countCross] = self.pop[pos].clone()
                countCross += 1

        return offspringPopulation
    
    
    def doCrossover(self):
        """ Arranges the crossover within the entire population and selects parents. Does not do actual crossing over of agents. """
        countCross = 0
        parents = []
        parent1 = None
        parent2 = None
        offspring = [None,None]
        offspringPopulation = [[] for i in range(cons.popSize)]
        
        for i in range(cons.popSize): #construct parent position list
            parents.append(i)
            
        p1 = -1         
        samp = Sampling(cons.popSize)
        for i in range(cons.popSize): #for each parent.
            if random() < cons.probCrossover:
                if p1 == -1:  #gets a first parent
                    p1 = samp.getSample()
                else:
                    p2 = samp.getSample()
                    pos1 = parents[p1]
                    pos2 = parents[p2]
                    parent1 = self.pop[pos1]
                    parent2 = self.pop[pos2]
                        
                    offspring = parent1.crossoverClassifiers(parent2)
                    offspringPopulation[countCross] = offspring[0]
                    countCross += 1
                    offspringPopulation[countCross] = offspring[1]
                    countCross += 1
                    p1 = -1
            else:
                pos = parents[samp.getSample()]
                offspringPopulation[countCross] = self.pop[pos].clone()
                countCross += 1
        if p1 != -1:
            pos = parents[p1]
            offspringPopulation[countCross] = self.pop[pos].clone()
            countCross += 1
                
        return offspringPopulation
    
    
    def doMutation(self, offspring):
        """ Just calls to mutate population of agents in GAssist_Agent. Also initiates the special stages. """
        for i in range(cons.popSize):
            if random() < cons.probMutationInd: #decides if a whole agent will undergo mutation.
                offspring[i].doMutation()
                

    def setModified(self):  
        """ Set entire population to not evaluated. """
        for i in range(cons.popSize):
            self.pop[i].agnPer.setIsEvaluated(False)
   
                
    def replacementPolicy(self, offspring, lastIteration):
        """ Handles Elitism and replacement of worst rules in the population by the best old ones """
        if lastIteration: #LAST ITERATION ONLY
            for i in range(self.numVersions):
                if self.bestAgents[i] != None:
                    self.evaluateClassifier(self.bestAgents[i]) #evaluate each best agent on some other subset of the data?
            
            set = []
            for i in range(cons.popSize):
                self.sortInsert(set,offspring[i])
            
            for i in range(self.numVersions):
                if self.bestAgents[i] != None:
                    self.sortInsert(set,self.bestAgents[i])

            for i in range(cons.popSize): #orders the offspring
                offspring[i] = set[i]

        else: #ALL OTHER ITERATIONS 
            previousVerUsed = False
            currVer = self.windows.getCurrentVersion() #Returns the current window id. (eg. 0,1,2,3) 

            if self.bestAgents[currVer] == None and currVer > 0: # If there is no best agent and there is at least one current Version so far
                previousVerUsed = True
                currVer -= 1 #use previous window.
            
            if self.bestAgents[currVer] != None: #elitism
                self.evaluateClassifier(self.bestAgents[currVer])
                worst = self.getWorst(offspring)  
                offspring[worst] = self.bestAgents[currVer].clone()  

            if not previousVerUsed: 
                prevVer = 0
                if currVer == 0:
                    prevVer = self.numVersions - 1  # i think this assumes that currVers can go to 0 as one of it's versions.
                else:
                    prevVer = currVer - 1
                
                if self.bestAgents[prevVer] != None:
                    self.evaluateClassifier(self.bestAgents[prevVer])
                    worst = self.getWorst(offspring)
                    offspring[worst] = self.bestAgents[prevVer].clone()
                    
        return offspring
                            
                    
    def sortInsert(self, set, cl):
        """ Sorts agents into a list according to comparisons. """
        for i in range(len(set)):
            if cl.compareToIndividual(set[i]):
                set.insert(i,cl)
                return
        set.append(cl)
                
                
    def getTracking(self):
        """ Get all the tracking values packaged as an array. """
        return [self.bestFitness, self.bestAccuracy, self.bestNumRules, self.bestAliverules, self.bestGenerality, self.averageFitness, self.averageAccuracy, self.averageNumRules, self.averageNumRulesUtils, self.averageGenerality] # 10 items


    def getIterationsSinceBest(self):
        """ Returns iterations since best """
        return self.iterationsSinceBest
    
    
# UTILITIES ************************************************************************************
    def getAverage(self, data):
        """ Get the average of a list of values. """
        ave = 0
        size = len(data)
        for i in range(size):
            ave += data[i]
        ave /= float(size)
        return ave
    
    
    def getDeviation(self, data):
        """ Get the standard deviation from a list of values. """
        ave = self.getAverage(data)
        dev = 0
        size = len(data)
        for i in range(size):
            val = data[i]
            dev += math.pow(val-ave,2.0)
            
        dev /= size
        return math.sqrt(dev)