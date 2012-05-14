#!/usr/bin/env python

#Import modules
from GALE_Constants import * 
from GALE_BatchPerformance import *
from random import *
from GALE_Environment import *
from copy import deepcopy

class GALEAgent:
    def __init__(self, e):  
        """ Initializes a single Agent Object... 
        Each agent has a vector of rules, each rule with a classification, and a performance value. """
        self.vector = []
        self.attributes = e.getNumAttributes()
        self.numClass = e.getNrActions()
        self.attLength = e.getAttributeLength()
        self.attCombos = e.getAttributeCombos()
        self.classList = [] #classList should be ['0','1']
        self.attList = []
        self.attCombList = []

        #Make a performance object for this agent.
        self.agnPer = GALEBatchPerformance(self.numClass)
        
        for i in range(self.attLength):
            self.attList.append(str(i))
        for i in range(self.numClass):
            self.classList.append(str(i))
        for i in range(self.attCombos):
            self.attCombList.append(str(i))
            
        ruleLength = self.attributes+1  #includes space for class

        # Generate the initial agent.
        for i in range(cons.agentInitWidth): # The initial number of rules in the individual agent
            tRuleC = []
            for j in range(self.attributes):    #Full data set coding - over all attributes - one bit per attribute
                if random.random() < cons.Wild:  #Chance that a position in the rule is 'don't care' (#)
                    tRuleC.append(cons.dontCare)
                else:
                    tRuleC.append(choice(self.attCombList)) #Otherwise randomly pick a possible attribute value (genotype).
            tRuleC.append(choice(self.classList))
            self.vector.append(tRuleC)
                    
        #Initialize tracking info for the agent.
        self.buildRuleTrack(len(self.vector))
          
                            
    def buildRuleTrack(self, vectLen):
        """ Initialize tracking values/lists for this agent. These will be updated for each training evaluation. """
        self.prune = []
        self.match = []        
        for i in range(vectLen):
            self.prune.append(0)
            self.match.append(0)              
                   
                            
    def resetPerformance(self):
        """ Resets the performance matrix to zeros, and resets prune and match values for each rule to zero. """
        self.agnPer.reset()
        for i in range(len(self.prune)):
            self.prune[i] = 0
            self.match[i] = 0
            
    
    def getPerformance(self):
        """ Returns the performance object for this agent. """
        return self.agnPer
    

    def agentMerge(self, agent2):
        """ Uses two point crossover to merge this individual with another that has been specified (agent1). """
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

        #Interpreting the resultant genetic material
        vec = []
        if iRT == 0:
            vec = self.translateArray(holdI, iRI)
        elif iRI ==0:
            vec = self.translateArray(holdT, iRT)
        else:
            if random.random() < cons.P_pickLargerX:
                if iRI < iRT:
                    vec = self.translateArray(holdT, iRT)
                else:
                    vec = self.translateArray(holdI, iRI)
            else:
                if iRI < iRT:
                    vec = self.translateArray(holdI, iRI)
                else:
                    vec = self.translateArray(holdT, iRT)
                    
        #print "Vector length pre-remove: " + str(len(vec)) #DEBUG
        vec = self.removeRepeatRules(vec) # Check for repeated rules and remove repeats
        #print "Vector length post-remove: " + str(len(vec)) #DEBUG
        
        #Clone self and update the clone parameters
        agnRes = self.clone()
        agnRes.vector  = vec
        agnRes.buildRuleTrack(len(agnRes.vector))
        return agnRes


    def removeRepeatRules(self, vect):
        """ Removes repeats of a given rule. """
        removeList = []
        for i in range(len(vect)-1): #vector length is getting smaller each time
            rule = vect[i]
            j = i+1
            while j < len(vect):
                if self.equalRules(vect[i], vect[j]):
                    if j not in removeList:
                        removeList.append(j)
                j += 1
        removeList.sort()
        removeList.reverse()
        for i in removeList:
            vect.pop(i)
        return vect
                

    def equalRules(self, rule1, rule2):
        """ Determines whether two rules are equivalent. """
        equal = False
        if rule1 == rule2:
            equal = True
        return equal
            
            
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
    
    
    def split(self):
        """ Clones and mutates this agent. """
        agnRes = self.clone()
        agnRes.agentMutate()
        return agnRes
        
    
    def agentMutate(self):
        """ Mutate genotype positions based on some mutation rate. """
        for i in range(len(self.vector)):
            for j in range(self.attributes):
                if random.random() < cons.SomaticMutationP:
                    if random.random() < cons.Wild:     #Chance that a position in the rule is 'don't care' (#)
                        self.vector[i][j] = cons.dontCare
                    else:
                        self.vector[i][j] = choice(self.attList)
            if random.random() < cons.SomaticMutationP:
                self.vector[i][self.attributes] = choice(self.classList)


    def clone(self):
        """ Makes a clone of this class object and returns a reference to it. """
        objClone = deepcopy(self)
        return objClone
    
    
    def getComplexity(self):
        """ Returns the number of rules comprising this agent (i.e complexity). """
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
        return wildTotal / float(total)
   
   
    def agentPrune(self):
        """ Prune the ruleset to get rid of useless rules """
        #Altered so that rules that have no match instead of no prune are pruned - Basically a less stringent implementation
        iVec = 0
        iNew = len(self.vector)
        iMax = len(self.vector)
        agnRes = None
        iaTmp = []
        iaMtc = []
        
        for i in range(iMax):
            if self.match[i] <= 0:      #New GALE implementation of Prune.
            #if self.prune[i] <= 0:    #Original GALE implementation of Prune!!!
                self.vector.pop(iVec)
                iNew -= 1
            else:
                iVec += 1
        
        #print "The vector after pruned is this long: "+str(len(self.vector)) #DEBUG
        if len(self.vector) == 0:   #If there are no rules left in the agent
            agnRes = None
        elif iMax == iNew:   # If all rules are still present
            agnRes = self
        else:
            #Remove useless things
            iVec = 0    
            iArr = 0 
            for i in range(len(self.vector)):
                iaTmp.append(0)
                iaMtc.append(0)   
            while iArr < iMax and iVec < len(iaTmp):
                if self.prune[iArr] > 0:
                    iaTmp[iVec] = self.prune[iArr]
                    iaMtc[iVec] = self.match[iArr]
                    iArr += 1
                    iVec += 1
                else:
                    iArr += 1
            self.prune = iaTmp
            self.match = iaMtc
            agnRes = self
        return agnRes
       
    
    def classify(self, instance):
        """ Classifies a given instance based on the rule vector """
        pClass = -1 #note how -1 i returned if no rule matches
        bFlag = True
        i = 0
        while bFlag and i < len(self.vector): #until a match is found within the rule set.
            rule = self.vector[i]
            if self.matched(rule[0:self.attributes],instance[0]): #The rule and instance should be the same length.
                self.match[i] += 1
                bFlag = False
                pClass = self.vector[i][self.attributes]
                if pClass == instance[1]:
                    self.prune[i] += 1
            i += 1
        
        self.agnPer.update(int(pClass), int(instance[1])) #This does the accuracy based classification, the above tracks match and prune
        return pClass

        
    def testClassify(self, instance):
        """  Classifies an instance without updating other parameters.  Returns -1 if no match, 0 for incorrect prediction and 1 for corect prediction. """
        pCorr = -1
        bFlag = True
        i = 0
        while bFlag and i < len(self.vector):
            rule = self.vector[i]
            if matched(rule[0:self.attributes],instance[0]):
                pClass = self.vector[i][self.attributes]
                bFlag = False
                if pClass == cClass:
                    pCorr = 1
                else:
                    pCorr = 0
            i += 1
        return pCorr
    
            
    def matched(self, rule, cState):
        """ Determines whether an instance is matched by a given rule """
        if len(cState) != len(rule):
            return False
        
        for i in range(len(rule)):
            if rule[i] != cons.dontCare and rule[i]!=cState[i]:
                return False;
            
        return True
    
    
    def getVector(self):
        """ Returns the agent's vector/list of rules. """
        return self.vector
    
    
    def getPrune(self):
        """ Returns prune - a list storing the number of times each rule in the agent matched and made a correct classification. """
        return self.prune
    
    
    def getMatch(self):
        """ Returns match - a list storing the number of times each rule matched an instance. 
        NOTE: each rule does not see all instances as the rule set is ordered, and once a preceeding rule matches an instance, all following rules will not get to see that instance. """
        return self.match
