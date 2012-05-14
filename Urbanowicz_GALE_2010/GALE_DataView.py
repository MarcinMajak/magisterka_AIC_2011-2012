#!/usr/bin/env python

class DataView:
    def __init__(self, dataset, iLb, iUb):
        """ Initialize the data set. """
        self.dataset = dataset
        self.iLowBound = iLb
        self.iUpBound = iUb
        self.iSize = iUb - iLb + 1
        
        
    def getInstance(self, pos):
        """ Returns an instance of the data set according to a specified position. """
        if pos >= 0 and pos < self.iSize:
            return self.dataset[pos+self.iLowBound]
        else:
            return None
    
    
    def getSize(self):
        """ Returns the size of the data set. """
        return self.iSize
