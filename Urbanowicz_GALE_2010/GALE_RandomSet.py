#!/usr/bin/env python

#Import modules
from random import *

class RandomSet:
    def __init__(self, dataset):
        """ Initialize the data set. """
        self.dataset = dataset
        
        
    def getInstance(self, pos): #No position is actually required.
        """ Returns a random instance of the data set. """
        place = randint(0,(self.getSize()-1))
        return self.dataset[place]
    
    
    def getSize(self):
        """ Returns the size of the data set. """
        return len(self.dataset)
    
