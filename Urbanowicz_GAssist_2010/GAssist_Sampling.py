#-------------------------------------------------------------------------------
# Name:        GAssist_Sampling.py
# Purpose:     A Python Implementation of GAssist
# Author:      Ryan Urbanowicz (ryanurbanowicz@gmail.com) - code primarily based on Java GALE authored by Jaume Bacardit <jaume.bacardit@nottingham.ac.uk>
# Created:     10/04/2009
# Updated:     2/22/2010
# Status:      FUNCTIONAL * FINAL * v.1.0
# Description: Handles sampling without replacement.
#-------------------------------------------------------------------------------
#!/usr/bin/env python

#Import modules
from random import *

class Sampling:
    def __init__(self, maxSize): 
        """  Initialize sampling objects """
        self.maxSize = maxSize
        self.sample = []
        self.num = 0
        self.initSampling()
        
        
    def initSampling(self):
        """ Initialize the sampling indexing list and maxSize parameter. """
        for i in range(0,self.maxSize):
            self.sample.append(i)
            self.num = self.maxSize
            
            
    def numSamplesLeft(self):
        """ Returns the number of samples left to draw from. """
        return self.num
    
    
    def getSample(self):
        """ Returns the next samples list position, and then removes that position from future consideration. """
        pos = randint(0,self.num-1)
        value = self.sample[pos]
        self.sample[pos] = self.sample[self.num-1]
        self.num -= 1
        if self.num == 0:
            self.initSampling()
        return value