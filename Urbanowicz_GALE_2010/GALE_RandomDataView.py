#!/usr/bin/env python

#Import modules
from random import *

class RandomDataView:
    def __init__(self, dataset, iLb, iUb):
        """ Initialize the data set. """
        self.dataset = dataset
        self.iLowBound = iLb
        self.iUpBound = iUb
        self.iSize = iUb - iLb + 1
        
        
    def getInstance(self, pos):
        """ Returns a random instance of the data set. """
        if pos >= 0 and pos < self.iSize:
            return self.dataset[randint(0,(self.getSize()-1))]
        else:
            return None
    
    #place = randint(0,(self.getSize()-1))
    def getSize(self):
        """ Returns the size of the data set. This may differ per cell using the pyramidal distribution"""
        return self.iSize
