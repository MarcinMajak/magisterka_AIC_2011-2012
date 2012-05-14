#!/usr/bin/env python

#Import modules
from GALE_Constants import *  
from GALE_Agent import *
from GALE_OrderedSet import *
from GALE_RandomSet import *
from GALE_DataView import *
from GALE_RandomDataView import *
from random import *
import copy
import math
from copy import deepcopy

class GALEBoard:    
    def __init__(self, xmax, ymax, brdDist, inSet, env, testSet):
        """ Initialize the 2D board learning environment """
        print "Initializing Board."
        #Define Board objects
        self.Xmax = xmax
        self.Ymax = ymax
        self.boardDistType = brdDist
        self.env = env

        self.bestAgent = None   #Holds/tracks the best agent found along the run.
        self.iBoard = 0   #Tracks the current board - the 3rd dimension value of agentBoard
        
        self.board = []  #The 2D board - self.board[b][x][y] where b is the board instance.
        self.agentPerformance = []
        
        #Mapping Variables
        self.instances = inSet
        self.testSet = testSet
        self.baAllIns = []
        self.osaMapping = []
        
        #Tracking Variables
        self.bestAccuracy = 0
        self.bestComplexity = 0
        self.bestGenerality = 0
        self.bestNoMatch = 0
        
        self.aveAccuracy = 0
        self.aveComplexity = 0
        self.aveGenerality = 0
        self.aveNoMatch = 0
        self.numAgents = 0
        
        #All of the below 2D arrays are referenced as self.variable[xmax][ymax]
        #Build framework for board
        for v in range(2):
            self.board.append([])
            for i in range(self.Xmax):
                self.board[v].append([])
                for j in range(self.Ymax):
                    self.board[v][i].append(None)
                            
        #Build framework for agentPerformance
        for i in range(self.Xmax):
            self.agentPerformance.append([])
            self.baAllIns.append([])
            self.osaMapping.append([])
            for j in range(self.Ymax):
                self.agentPerformance[i].append(None)  
                self.baAllIns[i].append(None)
                self.osaMapping[i].append(None)
                
        #Initialize the board (agents)
        i = 0
        iX = 0
        iY = 0
        initAgents = int(cons.iAgent * self.Xmax * self.Ymax)
        print "Making " + str(initAgents) + " agents."
        while i < int(initAgents):
            iX = randint(0,self.Xmax-1)
            iY = randint(0,self.Ymax-1)
            if self.board[self.iBoard][iX][iY] == None:
                self.board[self.iBoard][iX][iY] = GALEAgent(self.env)
                i += 1
                self.agentPerformance[iX][iY] = self.board[self.iBoard][iX][iY].getPerformance()
                
        #Initialize the best agent
        self.bestAgent = self.board[self.iBoard][iX][iY].clone() #Clone appears to be working, as does the agent initialization
        
        #Initialized the mapping of the data across the board
        self.arrangeMapping(self.instances, self.boardDistType)

        #Compute initial performance
        self.computePerformance()       
        print "Finish initial performance calculations." 
    
           
    def getBestAgent(self):
        """ Returns the best agent as an object. """
        return self.bestAgent
    
    
    def fullMappedAgents(self):
        """ Returns the full mapped agents. Subtle difference from original GALE - here we return list of agent references. """
        vec = []
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.baAllIns[i][j] and self.board[iBoard][i][j] != None:
                    vec.append(self.board[iBoard][i][j])
        return vec
                

    def evolveBoard(self):
        """ Runs a single evolution of the board. Directs the evolution of the board over the standard GALE evolution steps.  Each of these called methods calls for the 
        specified action to occur in every cell of the board """
        self.merge() #Crossover - no new agents produced
        self.split() #Mutates and creates a new agent
        self.computePerformance() # Does accuracy calculations
        self.survival() # Determines if given agent survives.
        self.computeStatistics() #updates all tracking parameters
            
            
    def split(self):
        """ Directs all cells to split. """
        thresh = float(0.0)
        for i in range(self.Xmax):  # From here on out in the original implementation they went Y, before X in the loops.
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j] != None:
                    thresh = self.agentPerformance[i][j].getScaledAccuracy() * cons.MaxSplitP
                    if random() < thresh: # Decides whether to split
                        #print "Split initiated." #DEBUG
                        iaSP = self.splitTargetPosition(i,j) #Finds an empty cell location within with to add the split agent
                        newAgent = self.board[self.iBoard][i][j].split()    # Calls the split on a specific agent
                        self.board[self.iBoard][iaSP[0]][iaSP[1]] = newAgent
                        self.agentPerformance[iaSP[0]][iaSP[1]] = newAgent.getPerformance()
                        
                
    def splitTargetPosition(self, i, j):  
        """ Finds an empty cell or worst cell location within with to place the new split agent. """
        iaPos = [-1, -1]
        iNei  = -1
        iTmp  = 0
        
        #Handles directionality with wrap around at edges of 2D board
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax
        
        iTmp = self.countNeighbours(iUp,iLeft)
        if self.board[self.iBoard][iUp][iLeft] == None and iTmp > iNei:
            iaPos[0] = iUp
            iaPos[1] = iLeft
            iNei     = iTmp
            
        iTmp = self.countNeighbours(iUp,j)        
        if self.board[self.iBoard][iUp][j] == None and iTmp > iNei:
            iaPos[0] = iUp
            iaPos[1] = j
            iNei     = iTmp;           
        
        iTmp = self.countNeighbours(iUp,iRight)
        if self.board[self.iBoard][iUp][iRight] == None and iTmp > iNei:
            iaPos[0] = iUp
            iaPos[1] = iRight
            iNei     = iTmp

        iTmp = self.countNeighbours(i,iLeft)
        if self.board[self.iBoard][i][iLeft] == None and iTmp > iNei:
            iaPos[0] = i
            iaPos[1] = iLeft
            iNei     = iTmp

        iTmp = self.countNeighbours(i,iRight)
        if self.board[self.iBoard][i][iRight] == None and iTmp > iNei:
            iaPos[0] = i
            iaPos[1] = iRight
            iNei     = iTmp

        iTmp = self.countNeighbours(iDown,iLeft)
        if self.board[self.iBoard][iDown][iLeft] == None and iTmp > iNei:        
            iaPos[0] = iDown
            iaPos[1] = iLeft
            iNei     = iTmp
            
        iTmp = self.countNeighbours(iDown,j)            
        if self.board[self.iBoard][iDown][j] == None and iTmp > iNei:            
            iaPos[0] = iDown
            iaPos[1] = j
            iNei     = iTmp      

        iTmp = self.countNeighbours(iDown,iRight)       
        if self.board[self.iBoard][iDown][iRight] == None and iTmp > iNei: 
            iaPos[0] = iDown
            iaPos[1] = iRight
            iNei     = iTmp          
        
        if iNei > -1:
            return iaPos
        else:   #if there is no free space available
            return self.worstNeighbour(i,j)


    def worstNeighbour(self, i, j):
        """ Returns the worst neighbor for the given cell. """
        #Handles directionality with wrap around at edges of 2D board
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax
        
        worst = float(1000.0)    
        iaRes = [iUp,iLeft]

        if self.board[self.iBoard][iUp][iLeft]!=None:
            if worst > self.agentPerformance[iUp][iLeft].getScaledAccuracy():
                worst = self.agentPerformance[iUp][iLeft].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = iLeft                
                
        if self.board[self.iBoard][iUp][j]!=None:
            if worst > self.agentPerformance[iUp][j].getScaledAccuracy():
                worst = self.agentPerformance[iUp][j].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = j                  
        
        if self.board[self.iBoard][iUp][iRight]!=None:
            if worst > self.agentPerformance[iUp][iRight].getScaledAccuracy():
                worst = self.agentPerformance[iUp][iRight].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = iRight 

        if self.board[self.iBoard][i][iLeft]!=None:
            if worst > self.agentPerformance[i][iLeft].getScaledAccuracy():
                worst = self.agentPerformance[i][iLeft].getScaledAccuracy()
                iaRes[0] = i
                iaRes[1] = iLeft

        if self.board[self.iBoard][i][iRight]!=None:
            if worst > self.agentPerformance[i][iRight].getScaledAccuracy():
                worst = self.agentPerformance[i][iRight].getScaledAccuracy()
                iaRes[0] = i
                iaRes[1] = iRight
                
        if self.board[self.iBoard][iDown][iLeft]!=None:
            if worst > self.agentPerformance[iDown][iLeft].getScaledAccuracy():
                worst = self.agentPerformance[iDown][iLeft].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = iLeft            
                
        if self.board[self.iBoard][iDown][j]!=None:
            if worst > self.agentPerformance[iDown][j].getScaledAccuracy():
                worst = self.agentPerformance[iDown][j].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = j                     
                
        if self.board[self.iBoard][iDown][iRight]!=None:
            if worst > self.agentPerformance[iDown][iRight].getScaledAccuracy():
                worst = self.agentPerformance[iDown][iRight].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = iRight 
            
        return iaRes
   
   
    def merge(self):
        """ Directs all cells to merge. """
        iaPos = [None,None]
        iNextBoard = (self.iBoard +1)%2 # At this point self.iBoard is always 0, and iNextboard is always 1
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j]!=None:
                    agn = self.crossingNeighbour(i,j) #Picks neighbor for merger
                    if agn != None and random() < cons.MergeP: #Decides whether to merge.
                        #print "Merge initiated"    #DEBUG
                        agnNew = self.board[self.iBoard][i][j].agentMerge(agn)
                        self.board[iNextBoard][i][j] = agnNew
                    else:
                        self.board[iNextBoard][i][j] = self.board[self.iBoard][i][j]
                else:
                    self.board[iNextBoard][i][j] = None
        
        #Switching the board            
        self.iBoard = iNextBoard  # first time - now it's 1
  
        #Update performance references
        self.updatePerformance()
        
        
    def crossingNeighbour(self, i, j):
        """ Returns the crossing neighbor - counts neighbors, and then randomly selects and returns one of those neighbors. """
        agnRes = None
        iNeigh = self.countNeighbours(i, j)
        if iNeigh > 0:
            iRndNei = randint(0,iNeigh)
            iaPos = self.getIthNeighbour(i,j,iRndNei)
            agnRes = self.board[self.iBoard][iaPos[0]][iaPos[1]]
        
        return agnRes
        
    
    def getIthNeighbour(self, i, j, iNei):
        """ Returns the ith neighbor """
        iaRes = [-1,-1]
        
        #Handles directionality with wrap around at edges of 2D board        
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax
        
        bDone = False
        
        if self.board[self.iBoard][iUp][iLeft] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iUp
                iaRes[1] = iLeft
                bDone = True
            else:
                iNei -= 1

        if self.board[self.iBoard][iUp][j] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iUp
                iaRes[1] = j
                bDone = True
            else:
                iNei -= 1        

        if self.board[self.iBoard][iUp][iRight] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iUp
                iaRes[1] = iRight
                bDone = True
            else:
                iNei -= 1         

        if self.board[self.iBoard][i][iLeft] != None and not bDone:
            if iNei == 0:
                iaRes[0] = i
                iaRes[1] = iLeft
                bDone = True
            else:
                iNei -= 1   
        
        if self.board[self.iBoard][i][iRight] != None and not bDone:
            if iNei == 0:
                iaRes[0] = i
                iaRes[1] = iRight
                bDone = True
            else:
                iNei -= 1 

        if self.board[self.iBoard][iDown][iLeft] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iDown
                iaRes[1] = iLeft
                bDone = True
            else:
                iNei -= 1         

        if self.board[self.iBoard][iDown][j] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iDown
                iaRes[1] = j
                bDone = True
            else:
                iNei -= 1  
        
        if self.board[self.iBoard][iDown][iRight] != None and not bDone:
            if iNei == 0:
                iaRes[0] = iDown
                iaRes[1] = iRight
                bDone = True
            else:
                iNei -= 1  

        return iaRes
        

    def survival(self):
        """ Determines survival of each agent. """
        iNei = None
        iNextBoard = (self.iBoard+1)%2
        fMean = float(0.0)
        fStdDev = float(0.0)
        fThreshold = float (0.0)
        deathcount = 0
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j] != None:
                    iNei = self.countNeighbours(i,j)
                    if iNei <= 1:
                        #Solitude
                        if random() < (1.0 - self.agentPerformance[i][j].getScaledAccuracy()):
                            self.board[iNextBoard][i][j] = None
                            #print "Agent Death via solitude"    #DEBUG
                            deathcount += 1
                        else:
                            self.board[iNextBoard][i][j] = self.board[self.iBoard][i][j]
                    elif iNei > 6:
                        #Crowded
                        iaPos = self.bestNeighbour(i,j)
                        self.board[iNextBoard][i][j] = self.board[self.iBoard][iaPos[0]][iaPos[1]].clone()
                    else:
                        #Steady
                        fMean = self.neighbourMean(i,j)
                        fStdDev = self.neighbourStdDev(i,j,fMean)
                        fKThresh = fMean + cons.kThreshold * fStdDev
                        if self.agentPerformance[i][j].getScaledAccuracy() < fKThresh:
                            self.board[iNextBoard][i][j] = None
                            #print "Agent Death via steady"    #DEBUG
                            deathcount += 1
                        else:
                            self.board[iNextBoard][i][j] = self.board[self.iBoard][i][j]   
                else:
                    self.board[iNextBoard][i][j] = None
        #print str(deathcount) +" agents eliminated this survival phase."    #DEBUG
        
        #Switching the board            
        self.iBoard = iNextBoard
  
        #Update performance references - doesn't recalculate accuracies
        self.updatePerformance()
    

    def updatePerformance(self):
        """ Updates the performance references. """
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j] != None:
                    self.agentPerformance[i][j] = self.board[self.iBoard][i][j].getPerformance()
                else:
                    self.agentPerformance[i][j] = None
 

    def countNeighbours(self, i, j):
        """ Count the number of neighbors for a given cell position. """
        #Handles directionality with wrap around at edges of 2D board        
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax        
        
        iRes = 0
        
        if self.board[self.iBoard][iUp][iLeft] != None:
            iRes += 1
        
        if self.board[self.iBoard][iUp][j] != None:
            iRes += 1
        
        if self.board[self.iBoard][iUp][iRight] != None:
            iRes += 1
        
        if self.board[self.iBoard][i][iLeft] != None:
            iRes += 1        

        if self.board[self.iBoard][i][iRight] != None:
            iRes += 1
        
        if self.board[self.iBoard][iDown][iLeft] != None:
            iRes += 1  

        if self.board[self.iBoard][iDown][j] != None:
            iRes += 1
        
        if self.board[self.iBoard][iDown][iRight] != None:
            iRes += 1  
        
        return iRes

        
    def bestNeighbour(self, i, j):
        """ Returns the position of the best neighbor for a given cell. """
        #Handles directionality with wrap around at edges of 2D board
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax
        
        best = float(-100.0)    
        iaRes = [iUp,iLeft]

        if self.board[self.iBoard][iUp][iLeft]!=None:
            if best < self.agentPerformance[iUp][iLeft].getScaledAccuracy():
                best = self.agentPerformance[iUp][iLeft].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = iLeft                
                
        if self.board[self.iBoard][iUp][j]!=None:
            if best < self.agentPerformance[iUp][j].getScaledAccuracy():
                best = self.agentPerformance[iUp][j].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = j                  
        
        if self.board[self.iBoard][iUp][iRight]!=None:
            if best < self.agentPerformance[iUp][iRight].getScaledAccuracy():
                best = self.agentPerformance[iUp][iRight].getScaledAccuracy()
                iaRes[0] = iUp
                iaRes[1] = iRight 

        if self.board[self.iBoard][i][iLeft]!=None:
            if best < self.agentPerformance[i][iLeft].getScaledAccuracy():
                best = self.agentPerformance[i][iLeft].getScaledAccuracy()
                iaRes[0] = i
                iaRes[1] = iLeft

        if self.board[self.iBoard][i][iRight]!=None:
            if best < self.agentPerformance[i][iRight].getScaledAccuracy():
                best = self.agentPerformance[i][iRight].getScaledAccuracy()
                iaRes[0] = i
                iaRes[1] = iRight
                
        if self.board[self.iBoard][iDown][iLeft]!=None:
            if best < self.agentPerformance[iDown][iLeft].getScaledAccuracy():
                best = self.agentPerformance[iDown][iLeft].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = iLeft            
                
        if self.board[self.iBoard][iDown][j]!=None:
            if best < self.agentPerformance[iDown][j].getScaledAccuracy():
                best = self.agentPerformance[iDown][j].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = j                     
                
        if self.board[self.iBoard][iDown][iRight]!=None:
            if best < self.agentPerformance[iDown][iRight].getScaledAccuracy():
                best = self.agentPerformance[iDown][iRight].getScaledAccuracy()
                iaRes[0] = iDown
                iaRes[1] = iRight 
            
        return iaRes


    def neighbourMean(self, i, j):
        """ Computes the neighbor mean evaluation of the neighborhood of the given cell. """
        #Handles directionality with wrap around at edges of 2D board
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax        
        
        iNei = 0
        fAcc = float(0.0)
        
        if self.board[self.iBoard][iUp][iLeft] != None:
            fAcc += self.agentPerformance[iUp][iLeft].getScaledAccuracy()
            iNei += 1

        if self.board[self.iBoard][iUp][j] != None:
            fAcc += self.agentPerformance[iUp][j].getScaledAccuracy()
            iNei += 1        

        if self.board[self.iBoard][iUp][iRight] != None:
            fAcc += self.agentPerformance[iUp][iRight].getScaledAccuracy()
            iNei += 1

        if self.board[self.iBoard][i][iLeft] != None:
            fAcc += self.agentPerformance[i][iLeft].getScaledAccuracy()
            iNei += 1 

        if self.board[self.iBoard][i][j] != None:
            fAcc += self.agentPerformance[i][j].getScaledAccuracy()
            iNei += 1

        if self.board[self.iBoard][i][iRight] != None:
            fAcc += self.agentPerformance[i][iRight].getScaledAccuracy()
            iNei += 1 

        if self.board[self.iBoard][iDown][iLeft] != None:
            fAcc += self.agentPerformance[iDown][iLeft].getScaledAccuracy()
            iNei += 1

        if self.board[self.iBoard][iDown][j] != None:
            fAcc += self.agentPerformance[iDown][j].getScaledAccuracy()
            iNei += 1 

        if self.board[self.iBoard][iDown][iRight] != None:
            fAcc += self.agentPerformance[iDown][iRight].getScaledAccuracy()
            iNei += 1 

        return fAcc/iNei


    def neighbourStdDev(self, i, j, fMean):
        """ Computes the standard deviation of the neighborhood scaled accuracy of the given cell. """
        #Handles directionality with wrap around at edges of 2D board
        iUp =   (i+1)%self.Ymax
        iDown = (i+self.Ymax - 1)%self.Ymax
        iRight =(j+1)%self.Xmax
        iLeft = (j+self.Xmax - 1)%self.Xmax           
        
        iNei = 0
        fAcc = float(0.0)
        fTmp = 0
        
        if self.board[self.iBoard][iUp][iLeft] != None:
            fTmp += self.agentPerformance[iUp][iLeft].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1

        if self.board[self.iBoard][iUp][j] != None:
            fTmp += self.agentPerformance[iUp][j].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1        

        if self.board[self.iBoard][iUp][iRight] != None:
            fTmp += self.agentPerformance[iUp][iRight].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1

        if self.board[self.iBoard][i][iLeft] != None:
            fTmp += self.agentPerformance[i][iLeft].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1 

        if self.board[self.iBoard][i][j] != None:
            fTmp += self.agentPerformance[i][j].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1

        if self.board[self.iBoard][i][iRight] != None:
            fTmp += self.agentPerformance[i][iRight].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1 

        if self.board[self.iBoard][iDown][iLeft] != None:
            fTmp += self.agentPerformance[iDown][iLeft].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1

        if self.board[self.iBoard][iDown][j] != None:
            fTmp += self.agentPerformance[iDown][j].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1 

        if self.board[self.iBoard][iDown][iRight] != None:
            fTmp += self.agentPerformance[iDown][iRight].getScaledAccuracy()
            fAcc += fTmp*fTmp
            iNei += 1                 

        return math.sqrt(fAcc/float(iNei))
                   
        
    def computeStatistics(self):
        """ Compute the statistics of the board. """
        #Compute Statistics for the Best Agent
        self.bestAccuracy = self.bestAgent.agnPer.getAccuracy()
        self.bestComplexity = self.bestAgent.getComplexity()
        self.bestGenerality = self.bestAgent.getGenerality()
        self.bestNoMatch = self.bestAgent.agnPer.getNoMatch()   
        
        #Compute average statistics for board
        self.numAgents = 0
        self.aveAccuracy = 0
        self.aveComplexity = 0
        self.aveGenerality = 0
        self.aveNoMatch = 0
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j] != None:
                    self.numAgents += 1
                    self.aveAccuracy += self.board[self.iBoard][i][j].agnPer.getAccuracy()
                    self.aveComplexity += self.board[self.iBoard][i][j].getComplexity()
                    self.aveGenerality += self.board[self.iBoard][i][j].getGenerality()
                    self.aveNoMatch += self.board[self.iBoard][i][j].agnPer.getNoMatch() 
        if self.numAgents == 0:
            #print "No Agents mapped!"    #DEBUG
            self.aveAccuracy = 0
            self.aveComplexity = 0
            self.aveGenerality = 0   
            self.aveNoMatch = 0
        else:
            self.aveAccuracy = self.aveAccuracy / float(self.numAgents)
            self.aveComplexity = self.aveComplexity / float(self.numAgents)
            self.aveGenerality = self.aveGenerality / float(self.numAgents)    
            self.aveNoMatch = self.aveNoMatch  / float(self.numAgents)
                    
                    
    def getTracking(self):
        """ Get all the tracking values packaged as an array. """
        return [self.numAgents, self.bestAccuracy, self.bestComplexity, self.bestNoMatch, self.bestGenerality, self.aveAccuracy, self.aveComplexity, self. aveNoMatch, self.aveGenerality]
        
        
    def arrangeMapping(self, os, strategy): 
        """ Arranges the data mapping over the board. """
        if str(strategy) == '0':
            self.uniformMapping(os)
        elif str(strategy) == '1':
            self.randomMapping(os)
        elif str(strategy) == '2' or '3':
            self.pyramidTailNumberMapping(os, strategy)
        else:
            print str(strategy) + " is not a valid mapping strategy"
            
            
    def computePerformance(self):
        """ Computes the number of samples covered by the individual. """
        iBestPosY  = -1
        iBestPosX  = -1
        fTmp       = 0
        fTmpCmplx  = 0
        fBestEval  = self.bestAgent.getPerformance().getAccuracy()
        fBestCmplx = self.bestAgent.getComplexity()
        bPrune = cons.Prune
        bTmp = False
        
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                if self.board[self.iBoard][i][j] != None:
                    self.board[self.iBoard][i][j].resetPerformance() #clears out performance, match and prune parameters
                    #Agent Performance Computation
                    for k in range(self.osaMapping[i][j].getSize()): # gets number of data instances
                        ins = self.osaMapping[i][j].getInstance(k)   # increments thru instances
                        self.board[self.iBoard][i][j].classify(ins)
                    
                    #Pruning unused things
                    if bPrune:
                        self.board[self.iBoard][i][j] = self.board[self.iBoard][i][j].agentPrune()
                        if self.board[self.iBoard][i][j] == None:
                            self.agentPerformance[i][j] = None
                        else:
                            self.agentPerformance[i][j] = self.board[self.iBoard][i][j].getPerformance()
                            
                    #Updating the best information
                    if self.board[self.iBoard][i][j] != None and self.baAllIns[i][j]: # The best agent can only be one with access to all data.
                        #print str(fTmp)+"\t"+str(fBestEval) #DEBUG
                        fTmp = self.agentPerformance[i][j].getAccuracy()
                        fTmpCmplx = self.board[self.iBoard][i][j].getComplexity()
                        if fBestEval < fTmp or (fBestEval == fTmp and fTmpCmplx < fBestCmplx):
                            fBestEval = fTmp #These are reset here in-case within this iteration there are even better agents.
                            fBestCmplx = fTmpCmplx
                            iBestPosX = i
                            iBestPosY = j
                                
        if iBestPosX != -1 and iBestPosY != -1 and self.board[self.iBoard][iBestPosX][iBestPosY] != None:
            #print "Best agent reset. "    #DEBUG
            self.bestAgent = self.board[self.iBoard][iBestPosX][iBestPosY].clone()
            
            
    def uniformMapping(self, os):
        """ Arranges the data mapping over the board uniformly. """
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                self.osaMapping[i][j] = OrderedSet(os)
                self.baAllIns[i][j] = True
                
                
    def randomMapping(self, os):
        """ Arranges the data mapping over the board with random sampling.  """
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                self.osaMapping[i][j] = RandomSet(os)
                self.baAllIns[i][j] = True
                
                
    def pyramidTailNumberMapping(self, os, strategy):
        """ Arranges the data mapping over the board with pyrimid mapping. """
        ismSize = len(os)
        iTmp = -1
        iEnd = -1
        
        iFirstSliceX  = self.Xmax/3
        iFirstSliceY  = self.Ymax/3
        iSecondSliceX = 2 * iFirstSliceX + (self.Xmax - 3 * iFirstSliceX)
        iSecondSliceY = 2 * iFirstSliceY + (self.Ymax - 3 * iFirstSliceY)

        fXStep = 1.0/float(iFirstSliceX + 1)
        fYStep = 1.0/float(iFirstSliceY + 1)
        
        for i in range(self.Xmax):
            for j in range(self.Ymax):
                self.baAllIns[i][j] = False
                if i < iFirstSliceX:
                    if j < iFirstSliceY:
                        if i < j:
                            iEnd = ismSize * fXStep * (i+1)
                        else:
                            iEnd = ismSize * fYStep * (j+1)
                    elif j < iSecondSliceY:
                        iEnd = ismSize * fXStep * (i+1)
                    
                    else:
                        if i < self.Ymax - j - 1:
                            iEnd = ismSize * fXStep * (i+1)
                        else:
                            iTmp = self.Ymax - j
                            iEnd = ismSize * fYStep * iTmp
                elif i < iSecondSliceX:
                    if j < iFirstSliceY:
                        iEnd = ismSize * fYStep * (j+1)
                    elif j < iSecondSliceY: # Defines the center slice where all data is included.
                        iEnd = ismSize - 1
                        if i == iFirstSliceX or j == iFirstSliceY or (i+1) == iSecondSliceX or (j+1) == iSecondSliceY:
                            self.baAllIns[i][j] = False
                        else:
                            self.baAllIns[i][j] = True
                    else:
                        iTmp = self.Ymax - j
                        iEnd = ismSize * fYStep * iTmp
                else:
                    if j < iFirstSliceY:  #note - on other side (of pyramid) iTmp handles decreasing iEnd
                        if self.Xmax - i - 1 < j:
                            iTmp = self.Xmax - i
                            iEnd = ismSize *fXStep * iTmp
                        else:
                            iEnd = ismSize * fYStep * (j+1)
                    elif j < iSecondSliceY:
                        iTmp = self.Xmax - i
                        iEnd = ismSize * fXStep * iTmp
                    else:
                        if i > j:
                            iTmp = self.Xmax - i
                            iEnd = ismSize * fXStep * iTmp
                        else:
                            itmp = self.Ymax - j
                            iEnd = ismSize *fYStep * iTmp
                            
                if str(strategy) == '2':
                    self.osaMapping[i][j] = DataView(os,0,int(iEnd))
                     
                else:
                    self.osaMapping[i][j] = RandomDataView(os,0,int(iEnd))
                
                
    def testPerform(self):
        """ Calculates the accuracy of the best agent on the testing data. """
        tmpAgent = self.bestAgent.clone()
        tmpAgent.resetPerformance()
        print len(self.testSet)
        print len(self.env.getTestSet())
        for i in range(len(self.testSet)):
            tmpAgent.classify(self.env.getTestSet()[i])

        return tmpAgent #tmpAgent.getPerformance()
