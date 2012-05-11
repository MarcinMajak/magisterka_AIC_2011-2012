#!/usr/bin/python2
# -*- coding: utf-8 -*- 

class FuzzyLogicClassifier(object):
    def __init__(self, filepath):
        print "Tworzę obiekt zwany FuzzyLogic"
        self.data = []
        self.label = []
        self.filepath = filepath

    def readData(self):
        try:
            fd = open(self.filepath, 'r')
            lines = fd.readlines()
            self.data = [map(float, x.strip().split(',')[1:]) for x in lines]
            self.label = [map(float, x.strip().split(',')[0]) for x in lines]
        except (ValueError, IOError):
            pass

    def getNumberOfAttributes(self):
        return len(self.data[0]) if len(self.data) else 0
    
    def sizeOfData(self):
        return len(self.data)

if __name__ == '__main__':
    fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    fuzzy.readData()
    print fuzzy.getNumberOfAttributes()
    print fuzzy.sizeOfData()
