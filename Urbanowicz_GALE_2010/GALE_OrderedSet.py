#!/usr/bin/env python

class OrderedSet:
    def __init__(self, dataset):
        """ Initialize the data set. """
        self.dataset = dataset
        
        
    def getInstance(self, pos):
        """ Returns an instance of the data set according to a specified position. """
        return self.dataset[pos]
    
    
    def getSize(self):
        """ Returns the size of the data set. """
        return len(self.dataset)
    
