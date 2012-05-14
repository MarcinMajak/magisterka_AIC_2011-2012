#!/usr/bin/env python

#Import modules
import random
import sys

class GALE_Environment:  
    def __init__(self, inFileString, testFileString, attLength): 
        """ Initialize data set objects. """
        self.nrActions = 2
        self.headerList = []
        self.numAttributes = 0  # Saves the number of attributes in the input file.
        self.numSamples = 0  
        self.attributeLength = attLength
        self.attributeCombos = 3
        self.testFileString = testFileString        
        self.classPosition = 0

        #Final data objects.
        self.fTrainData = self.formatData(inFileString,True)
        self.fTestData = self.formatData(testFileString,False)
        print len(self.fTrainData)
        print len(self.fTestData)
        
    def formatData(self, dataset, train):
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
            datasetList = []
            f = open(dataset, 'r')
            self.headerList = f.readline().rstrip('\n').split('\t')   #strip off first row
            for line in f:
                lineList = line.strip('\n').split('\t')
                datasetList.append(lineList)
            f.close()
            self.numAttributes = len(self.headerList) - 1 # subtract 1 to account for the class column
            self.classPosition = len(self.headerList) - 1    # Could be altered to look for "class" header
            if train:
                self.numSamples = len(datasetList)  
            print len(datasetList)
            
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
        for i in range(len(datasetList)):  # for each column - one for the attribute data and one for the class
            formatedDataset.append([])
        for i in range(len(datasetList)):
            formatedDataset[i] = [' ', ' ']
             
        # Fill in the matrix built above with the binary attribute encoding and the binary class value    
        for line in range(len(datasetList)):
            codeList = []
            for att in range(self.numAttributes):
                if datasetList[line][att] == '0': #might need to be double checked /think thru
                    for j in range(self.attributeLength):
                        codeList.append(SNP0[j])

                if datasetList[line][att] == '1':
                    for j in range(self.attributeLength):
                        codeList.append(SNP1[j])

                if datasetList[line][att] == '2':
                    for j in range(self.attributeLength):
                        codeList.append(SNP2[j])
            formatedDataset[line][0] = codeList
            formatedDataset[line][1] = datasetList[line][self.classPosition]                         

        from random import shuffle
        shuffle(formatedDataset) 
        print len(formatedDataset) 
        return formatedDataset

    def getNrActions(self):
        """ Returns the number of possible actions.  In the GH problem there are two classifications possible. """
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
