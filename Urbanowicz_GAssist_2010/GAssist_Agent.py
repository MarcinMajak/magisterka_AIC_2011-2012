#-------------------------------------------------------------------------------
# Name:        GAssist_Agent.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Initializes agent and holds methods to manipulate a single agent within the population.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Constants import * 
from GAssist_Performance import *
from random import *
from GAssist_Environment import *
from copy import deepcopy
import math

class GAssistAgent:
    def __init__(self, e, smart, cw):  #classList should be ['0','1']
        """ Initializes a single Agent Object... handles regular or smart initialization.
        Each agent has a vector of rules, each rule with a classification, and a performance value. """
        #Major agent Architecture
        self.vector = [] #the vector of rules/classifiers that make up this agent.  The default rule and class is implied, but not included in this vector.
        self.defaultClass = None #defines the default class for this agent/individual
        
        #Required data set information
        self.attributes = e.getNumAttributes()
        self.numClass = e.getNrActions()
        self.attLength = e.getAttributeLength()
        self.attCombos = e.getAttributeCombos()
        self.classList = []
        self.attList = []
        self.attCombList = [] 
        self.env = e
           
        for i in range(self.attLength):
            self.attList.append(str(i))
        for i in range(self.numClass):
            self.classList.append(str(i))
        for i in range(self.attCombos):
            self.attCombList.append(str(i))  
        
        self.classCounts = e.getClassCounts()
        self.instancesByClass = e.getInstancesByClass()
        
        #Make a performance object for this agent.
        self.agnPer = GAssistPerformance(self.numClass)
        
        if cons.defaultClass == "auto": #Sets the default class for this agent. auto default class has further implications later on in selection/crossover etc.
            self.defaultClass = randint(0,self.numClass-1)
        elif cons.defaultClass == "disabled":
            self.defaultClass = None
        else:
            self.defaultClass = cons.defaultClass
   
        # AGENT INITIALIZATION
        ruleLength = self.attributes+1  #includes space for class
        for i in range(cons.agentInitWidth): # An initial number of rules in the individual agent
            tRuleC = [] #holds the rule being built in a list
            ins = None
            if smart:  #Rules will match instances
                ins = self.getInstanceInit(cw)  #CW - class wise balance to probablility of selection.
                
            if ins == None: # If there is no smart initialization do random initialization   
                for j in range(self.attributes):
                    if random.random() < cons.wild:
                        tRuleC.append(cons.dontCare)
                    else:
                        tRuleC.append(choice(self.attCombList)) 
                        
                tempClass = randint(0,self.numClass-1)
                while cons.defaultClass != "disabled" and tempClass == self.defaultClass:
                    tempClass = randint(0,self.numClass-1)  
                if self.defaultClass == tempClass and cons.defaultClass != "disabled":
                    print "error"
                tRuleC.append(tempClass)
                self.vector.append(tRuleC)  
            else:
                for j in range(self.attributes):
                    if random.random() < cons.wild:
                        tRuleC.append(cons.dontCare)
                    else:
                        tRuleC.append(ins[0][j]) 
                         
                tRuleC.append(ins[1])  #uses the same class as the instance passed.
                self.vector.append(tRuleC)  
                             
        # MDL 
        self.theoryLength = 0.0
        self.exceptionsLength = 0.0
                
                
    def getInstanceInit(self, cw):
        """ Gets an instance for smart initialization """
        forbiddenClass = None
        if self.defaultClass != "disabled":
            forbiddenClass = self.defaultClass
        else:
            forbiddenClass = self.numClass  # value not possible.

        if cw: #class wise selection
            targetClass = forbiddenClass
            while targetClass == forbiddenClass or len(self.env.getInstancesByClass()[targetClass]) == 0:  #Possible but very unlikely infinate loop possible here.
                targetClass = randint(0, self.numClass-1)
            pos = self.env.getSamplesOfClasses()[targetClass].getSample()
            return self.env.getTrainSet()[self.env.getInstancesByClass()[targetClass][pos]]
        
        #Handles the situation where the class selection in initialization is not equal by class, but equal by instance. - Regular smart initialization.
        count = []
        for i in range(self.numClass):
            count.append(0)
        total = 0
        for j in range(self.numClass):
            if j == forbiddenClass:
                count[j] = 0
            else:
                count[j] = self.env.getSamplesOfClasses()[j].numSamplesLeft()
                total += count[j]
            
        pos = randint(0, total-1)  #ERROR !!!!!!!!!! total-1 = 0 apparently
        acum = 0
        for i in range(len(count)):
            acum += count[i]
            if pos < acum:
                inst = self.env.getSamplesOfClasses()[j].getSample()
                return self.env.getTrainSet()[self.env.getInstancesByClass()[j][inst]]
            
        
    def getNumRules(self):
        """ Returns the number of rules that make up this agent. """
        if cons.defaultClass != "disabled":
            return (len(self.vector)+1)
        return len(self.vector)
          
                         
    def classify(self, instance):
        """ Determines whether an instance is matched by any rule in the agent and returns the predicted class, or the default class """
        #nA = self.attributes
        noMatch = True
        i = 0
        while noMatch and i < len(self.vector): #stops looking for match when a match is found or when it has gone through all the rules.
            rule = self.vector[i]
            if self.match(rule[0:self.attributes],instance[0]):
                self.agnPer.positionRuleMatch = i
                return self.vector[i][self.attributes]
            i += 1
        if cons.defaultClass == "disabled":
            return -1
        else:
            self.agnPer.positionRuleMatch = len(self.vector)
            return self.defaultClass
        

    def match(self, rule, cState):
        """ Determines whether an instance is matched by a given rule """
        if len(cState) != len(rule):
            print "Rule length does not match instance length." #Debugging
            return False
        
        for i in range(len(rule)):
            if rule[i] != cons.dontCare and rule[i]!=cState[i]:
                return False;
            
        return True   
        
                
    def computeTheoryLength(self):
        """ Used in the MDL fitness calculations - directly based on the rule representation used. """
        numValues = self.env.getAttributeCombos() #this is specific to our data sets, where all attributes have 3 possible values.
        base = 2
        for i in range(len(self.vector)): # for all rules
            if self.agnPer.getActivationsOfRule(i) > 0:
                ruleTL = 0.0
                for j in range(self.env.getNumAttributes()):  #Over each attribute
                    if self.vector[i][j] == cons.dontCare:
                        ruleTL += 1.0
                    else:
                        ruleTL += (1.0 + math.log(numValues, base))
                self.theoryLength += ruleTL # see thesis pg 208-215 or so. for XCS version of this.
                
        if cons.defaultClass != "disabled": #The completely general default rule is included in theory length calculation (and effects the overall fitness)
            self.theoryLength += 0.00000001
            
        return self.theoryLength


    def getTheoryLength(self):
        """ Returns theory length."""
        return self.theoryLength
    
    
    def setExceptionsLength(self, el):
        """ Sets the exception length used by MDL """
        self.exceptionsLength = el
        
        
    def getExceptionsLength(self):
        """ Return exceptions length. """
        return self.exceptionsLength
        
        
    def deleteRules(self, whichRules): #whichRules is a list of rule positions
        """ Deletes rules from the agent which are not active (whichRules). """
        numRules = len(self.vector)
        rulesToDelete = len(whichRules)
        if numRules == 1 or rulesToDelete == 0:
            return

        if whichRules[rulesToDelete-1] == numRules: #Safety net - makes sure that the last entry isn't one too big. - Delete this part if it checks out. trying to delete a rule that doesn't exist
            print "Shouldn't be here! - DeleteRules"
            print whichRules
            print "Number of rules in agent = " + str(numRules)
            whichRules.pop(rulesToDelete -1)  # This is a quick fix to this odd problem - I should track this problem down still.
            rulesToDelete -= 1

        whichRules.reverse()
        for i in whichRules: #counts down
            self.vector.pop(i)  #ERROR HERE WHEN wild is set high!


    def getRealNumRules(self):
        """ Returns the number of rules that make up this agent. """
        if cons.defaultClass != "disabled":
            return (len(self.vector) + 1)
        return len(self.vector)

  
    def getGenerality(self):
        """ Returns the average generality of the rules making up this agent. """
        total = 0
        wildTotal = 0
        for i in range(len(self.vector)):
            for j in range(len(self.vector[i])):
                if self.vector[i][j] == cons.dontCare:
                    total += 1
                    wildTotal += 1
                else:
                    total += 1
        if total == 0:
            print "Empty vector: Generality calculated using cons.wild."
            return cons.wild
        else:
            return wildTotal / float(total)
  
  
    def compareToIndividual(self, ind):
        """ This function returns true if this individual is better than the the individual/agent 
        passed as a parameter. This comparison can be based on accuracy or a combination of accuracy and size """
        
        l1 = len(self.vector) #Length of first agent - number of rules
        l2 = len(ind.vector) #Length of second agent - number of rules
        f1 = self.agnPer.getFitness()  #Fitness of first agent
        f2 = ind.agnPer.getFitness()  #Fitness of second agent
        
        if cons.doHierarchicalSelection:
            if math.fabs(f1-f2) <= cons.hierarchicalSelectionThreshold:
                if l1 < l2:
                    return True
                if l1 > l2:
                    return False
        
        if cons.useMDL == False:
            if f1 > f2:
                return True
            if f1 < f2:
                return False
            if random.random() < 0.5:
                return True
            return False
        
        if f1 < f2:
            return True
        if f1 > f2:
            return False
        if random.random() < 0.5:
            return True
        return False
            
            
    def clone(self):
        """ Makes a clone of this class object and returns a reference to it. """
        objClone = deepcopy(self)
        return objClone


    def getNiche(self):
        """ Return the default class - also represents the agent niche. """
        return self.defaultClass
    
    
    def crossoverClassifiers(self, agent2):
        """ Uses two point crossover to merge this agent with another which has been specified (agent2)"""        
        sepA1 = random.randint(0,self.attributes)
        sepA2 = random.randint(0,agent2.attributes)

        #Size of Agents
        lenAgT = len(self.vector)
        lenAgI = len(agent2.vector)
        
        #Cut points for instances of this(T) agent
        sepTIst1 = randint(0,lenAgT-1)
        sepTIst2 = randint(0,lenAgT-1)
        if sepTIst2 < sepTIst1:
            tmp = sepTIst1
            sepTIst1 = sepTIst2
            sepTIst2 = tmp  
        
        #Cut points for instances of input(I) agent
        sepIIst1 = randint(0,lenAgI-1)
        sepIIst2 = randint(0,lenAgI-1)
        if sepIIst2 < sepIIst1:
            tmp = sepIIst1
            sepIIst1 = sepIIst2
            sepIIst2 = tmp  

        if sepTIst1 == sepTIst2 or sepIIst1 == sepIIst2:
            if sepA2 < sepA1:
                tmp = sepA1
                sepA1 = sepA2
                sepA2 = tmp                      
            
        #Prepare crossover structures
        ruleLen = self.attributes + 1    # The length of a given rule
        maxLen = lenAgT * ruleLen + lenAgI * ruleLen #The maximum combined length of the two strings
        condenseT = [] #stores the whole agent string
        condenseI = [] #stores the whole agent string
        holdT = [] #stores the newly formed crossover string
        holdI = [] #stores the newly formed crossover string

        for i in range(lenAgT):
            for j in range(ruleLen):
                condenseT.append(self.vector[i][j]) # need to reincorporate the class of the rule for the recoding - have to alter the other end as well
   
        for i in range(lenAgI):
            for j in range(ruleLen):
                condenseI.append(agent2.vector[i][j])

        for i in range(maxLen):
            holdT.append(None)
            holdI.append(None)
            
        #Set cross points
        XT1 = ruleLen * sepTIst1 + sepA1
        XT2 = ruleLen * sepTIst2 + sepA2
        XI1 = ruleLen * sepIIst1 + sepA1
        XI2 = ruleLen * sepIIst2 + sepA2
        #print str(XT1)+"\t" + str(XT2)+"\t" + str(XI1)+"\t" + str(XI2) #DEBUG
        
        #Exchange materials  ******************************    
        iT = 0
        iI = 0
        iRT = 0
        iRI = 0  
        
        #Copying the head of the genes  
        while iT < XT1:
            holdT[iRT] = condenseT[iT]
            iRT += 1
            iT += 1            
        while iI < XI1:
            holdI[iRI] = condenseI[iI]
            iRI += 1
            iI += 1

        #Swapping genes   (exchanges the middle portion of the string)
        while iI < XI2:
            holdT[iRT] = condenseI[iI]
            iRT += 1
            iI += 1
        while iT < XT2:
            holdI[iRI] = condenseT[iT]    
            iRI += 1
            iT += 1
                    
        #Copying the tail of the genes      
        while iT < ruleLen*lenAgT:
            holdT[iRT] = condenseT[iT]
            iT += 1
            iRT += 1
        while iI < ruleLen*lenAgI:
            holdI[iRI] = condenseI[iI]
            iI += 1
            iRI += 1
        
        #Interpreting the resultant genetic material   #NOTE: instead here We want both offspring, iRT and iRI
        vec1 = self.translateArray(holdI, iRI)
        vec2 = self.translateArray(holdT, iRT)
        
        #Clone agent objects to become offspring.
        off1 = self.clone()
        off2 = agent2.clone()
        
        #Make the vectors for the new clones that of the new offspring.
        off1.vector = vec1
        off2.vector = vec2
        
        newOffspring = [None,None] #list with both agent offspring as objects in list.
        newOffspring[0] = off1
        newOffspring[1] = off2
        
        return newOffspring
    
    
    def translateArray(self, array, strLen):
        """ Converts the genetic individual array string into the vector representation. """
        vect = []
        ruleLen = self.attributes + 1
        numRules = strLen/ruleLen
        for i in range(numRules):
            tRule = []
            for j in range(ruleLen):
                tRule.append(array[i*ruleLen+j])
            vect.append(tRule)
        return vect
    

    def doMutation(self):  
        """ Mutates rules of the given agent. """

        mutateClass = False #Is the class gonna be mutated or the condition?
        if random.random() < 0.1:
            mutateClass = True
            
        whichRule = randint(0, len(self.vector) - 1)  #picks rule to mutate
        if mutateClass:
            oldValue = self.vector[whichRule][self.attributes]
            newValue = randint(0, self.numClass - 1) #new class
            if self.numClass < 3 and cons.defaultClass != "disabled":
                while newValue == self.defaultClass:
                    newValue = randint(0, self.numClass - 1)
            else:    
                while newValue == oldValue and (cons.defaultClass != "disabled" and newValue == self.defaultClass):
                    newValue = randint(0, self.numClass - 1)
                    
            self.vector[whichRule][self.attributes] = newValue
        else: #mutate some position in the condition
            whichAtt = randint(0, self.attributes - 1)
            oldValue = self.vector[whichRule][whichAtt]
            newValue = randint(0, self.attCombos - 1) #new class
            while oldValue == newValue:
                newValue = randint(0, self.attCombos - 1) #new class
            self.vector[whichRule][whichAtt] = newValue
                
                
    def testEvaluateAgent(self, instanceWindow, globalMDL):
        """  Evaluates and defines stats for the best agent on testing data. """
        self.agnPer.resetPerformance(self.getNumRules())
        for j in range(len(instanceWindow)): #for each instance in the window
            real = instanceWindow[j][1] 
            predicted = self.classify(instanceWindow[j])  
            self.agnPer.addPrediction(predicted,real)

        self.agnPer.calculatePerformance(globalMDL, self)  #CRAAAAAP - self.globalMDL, ind
        accuracy = self.agnPer.getAccuracy()    
        return accuracy 


    def getVector(self):
        """ Returns the agent's vector/list of rules. """
        return self.vector
    
    
    def defaultClassCheck(self):
        """ For debugging"""
        for i in range(len(self.vector)):
            if cons.defaultClass != "disabled" and self.vector[i][self.attributes] == self.defaultClass:
                print "ERROR:  Agent rule has specified the defaultClass."

        