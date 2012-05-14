#-------------------------------------------------------------------------------
# Name:        GAssist_Environment.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Formats and handles 'gh' genetic heterogeneity snp data sets with discrete SNP attributes, and discrete case/control status.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from GAssist_Sampling import *
import random
import sys

class GAssist_Environment:  
    def __init__(self, inFileString, testFileString, attLength, smartInit): 
        """ Initialize data set objects. """
        self.nrActions = 2
        self.headerList = []
        self.datasetList = []
        self.numAttributes = 0  # Saves the number of attributes in the input file.
        self.numSamples = 0  
        self.attributeLength = attLength
        self.attributeCombos = 3
        self.testFileString = testFileString        
        self.classPosition = 0
        
        #Initialization
        self.doSmartInit = False
        if smartInit == "smart" or smartInit == "cw":
            self.doSmartInit = True
            
        #Rule Initialization Elements Needed
        self.classCounts = [0,0] #number of instances of each class
        self.instancesByClass = [[],[]] # organizes instances by class
        self.samplesOfClasses = []
        
        #Final data objects.
        self.fTrainData = self.formatData(inFileString, True)
        self.datasetList = []  #check this in other LCS VERSIONSS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.classCounts = [0,0] #number of instances of each class
        #self.instancesByClass = [[],[]]
        #self.samplesOfClasses = []
        self.fTestData = self.formatData(testFileString, False)
        

    def formatData(self, dataset, training):
        print "Constructing Formated Dataset"
        SNP0 = []
        SNP1 = []
        SNP2 = []
        
        if self.attributeLength==3:
            SNP0 = ['0','0','1']
            SNP1 = ['0','1','0']
            SNP2 = ['1','0','0']
        elif self.attributeLength==2:
            SNP0 = ['0','0']
            SNP1 = ['0','1']
            SNP2 = ['1','0']  
        elif self.attributeLength==1:
            #print "Use Direct Coding"
            SNP0 = ['0']
            SNP1 = ['1']
            SNP2 = ['2']  
        else:
            print "Coding Length out of bounds!"    

        #*******************Initial file handling**********************************************************
        try:       
            f = open(dataset, 'r')
            self.headerList = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                self.datasetList.append(lineList)
            f.close()
            self.numAttributes = len(self.headerList) - 1 # subtract 1 to account for the class column
            self.classPosition = len(self.headerList) - 1    # Could be altered to look for "class" header
            self.numSamples = len(self.datasetList)    
       
        except IOError, (errno, strerror):
            print ("Could not Read File!")
            print ("I/O error(%s): %s" % (errno, strerror))
            raise
        except ValueError:
            print ("Could not convert data to an integer.")
            raise
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise
        #**************************************************************************************************
        
        formatedDataset = []
        # Build empty matrix for formated data where each attribute gets it's own position in a list + class is at last position.
        for i in range(self.numSamples):  # for each column - one for the attribute data and one for the class
            formatedDataset.append([])
        for i in range(self.numSamples):
            formatedDataset[i] = [' ', ' ']
             
        # Fill in the matrix built above with the binary attribute encoding and the binary class value    
        for line in range(len(self.datasetList)):
            codeList = []
            for att in range(self.numAttributes):
                if self.datasetList[line][att] == '0': #might need to be double checked /think thru
                    for j in range(self.attributeLength):
                        codeList.append(SNP0[j])

                if self.datasetList[line][att] == '1':
                    for j in range(self.attributeLength):
                        codeList.append(SNP1[j])

                if self.datasetList[line][att] == '2':
                    for j in range(self.attributeLength):
                        codeList.append(SNP2[j])
            formatedDataset[line][0] = codeList
            formatedDataset[line][1] = self.datasetList[line][self.classPosition]      

            if self.doSmartInit and training:    #only make this effort if smart initialization is requested.
                #Further data set characterization 
                if self.datasetList[line][self.classPosition] == '0':
                    self.classCounts[0] += 1
                    self.instancesByClass[0].append(line) # Makes a nested list of references to where in formatedDataset the instance is located.
                elif self.datasetList[line][self.classPosition] == '1':
                    self.classCounts[1] += 1
                    self.instancesByClass[1].append(line) # Makes a nested list of references to where in formatedDataset the instance is located.
                else:
                    print "Data set class not recognized!"

        #Used later in sampling for smart initialization
        if self.doSmartInit and training:
            for i in range(self.nrActions): #for each class
                self.samplesOfClasses.append(Sampling(self.classCounts[i])) 

        return formatedDataset


    def getNrActions(self):
        """ Returns the number of possible actions/classes.  In the GH problem there are two classifications possible. """
        return self.nrActions


    def getNrSamples(self):
        """ Returns the number of samples in the data set being examined. """        
        return self.numSamples
    
    
    def getHeaderList(self):
        """ Returns the header text of attribute names. """
        return self.headerList
    
    
    def getAttributeLength(self):
        """ Returns the coding length (i.e. bitlength of this representation).  
        Arbitrary in this implementation, as it is set up to accomidate only a bit length of 1. """
        return self.attributeLength
    
    
    def getNumAttributes(self):
        """ Returns the number of attributes in the data set. """
        return self.numAttributes
    
    
    def getTrainSet(self):
        """ Returns the formated training set. """
        return self.fTrainData
    
    
    def getTestSet(self):
        """ Returns the formated testing set. """
        return self.fTestData
    
    
    def getAttributeCombos(self):
        """ Returns the possible number of attribute values. """
        return self.attributeCombos
    
    
    def getClassCounts(self):
        """ Returns a list with the number of instances representing each class. """
        return self.classCounts
    
    
    def getInstancesByClass(self):
        """ Returns a data set list organized by class. """
        return self.instancesByClass
    
    
    def getSamplesOfClasses(self):
        """ Returns the class sampling list """
        return self.samplesOfClasses