#!/usr/bin/python
# -*- coding: utf-8 -*- 

"""
This file contains basic classifiers definition
for comaprison purposes. They are trained and tested
on the same data as rough sets algorithm and fuzzy logic
"""

import mlpy
import sys
from consts import *

class BasicClassifiers(object):
    
    def __init__(self, debug=False):
        self.DEBUG = debug
    
    def read_data(self, filepath, label_is_last=True):
        """
        label_is_last indicates where label of the class is located.
        If it is True then it is the last column, otherwise it is the first
        """
        try:
            fd = open(filepath, 'r')
            lines = fd.readlines()
            if not label_is_last:
                self.data = [map(float, x.strip().split(',')[1:]) for x in lines]
                self.label = [map(float, x.strip().split(',')[0]) for x in lines]
            else:
                self.data = [map(float, x.strip().split(',')[:-1]) for x in lines]
                self.label = [map(float, x.strip().split(',')[-1]) for x in lines]
            fd.close()
            return True
        except (ValueError, IOError):
            return False
        
    def prepare_data(self, k_fold_number):
        """
        """
        self.k_fold_number = k_fold_number 
        self.__prepare_validation()
        
    def k_fold_cross_validation(self, k):
        """
        """
        self.current_k = k
        counter = self.size_of_data()/self.k_fold_number
        if k < 0 or counter == 0 or k*counter > self.size_of_data():
            return False
        
        self.training_data = []
        self.training_label = []
        self.testing_data = []
        self.testing_label = []
        
        for i in range(self.k_fold_number):
            if i == k:
                for obj in self.validation_container[k]:
                    self.testing_label.append(obj[0])
                    self.testing_data.append(obj[1])
            else:
                for obj in self.validation_container[i]:
                    self.training_label.append(obj[0])
                    self.training_data.append(obj[1])
        # for secure reason check if testing + learning is equal to data
        assert(len(self.training_data) + len(self.testing_data) == len(self.data))
        return True
    
    def get_number_of_attributes(self):
        """
        returns the number of attributes in available data
        """
        return len(self.data[0]) if len(self.data) else 0

    def size_of_data(self):
        """
        returns number of patterns 
        """
        return len(self.data)
    
    def __prepare_validation(self):
        my_dict = {}
        for i in range(0, len(self.data)):
            if not self.label[i][0] in my_dict.keys():
                my_dict[self.label[i][0]] = []
            my_dict[self.label[i][0]].append(self.data[i])
        
        self.validation_container = {}
        _work = True
        while _work:
            for i in range(self.k_fold_number):
                if not i in self.validation_container.keys():
                    self.validation_container[i] = []
                _sum = 0
                for _class in my_dict.keys():
                    if len(my_dict[_class]):
                        self.validation_container[i].append([_class, my_dict[_class].pop()])
                    _sum = _sum + len(my_dict[_class])
                if _sum == 0:
                    _work = False
                    break    

if __name__ == '__main__':
    DEBUG = False
    result_file = 'results/basic_classifiers.csv'
    try:
        fd = open(result_file, 'w')
    except IOError:
        print "Wrong path for results file"
        sys.exit(1)
    
    for d in range(len(datasets)):
        for k in range(K_FOLD_NUMBER):
            basic = BasicClassifiers(debug=DEBUG)
            filename = 'datasets/%s' % datasets[d][0]
            if basic.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1])) == False:
                print "Error with opening the file. Probably you have given wrong path"
                sys.exit(1)
            basic.prepare_data(k_fold_number=K_FOLD_NUMBER)
            basic.k_fold_cross_validation(k=k)
            
            size = len(basic.testing_label)
            
            ldac = mlpy.LDAC()
            ldac.learn(basic.training_data, basic.training_label)
            classified = 0
            for i in range(len(basic.testing_label)):
                if (int)(basic.testing_label[i]) == (int)(ldac.pred(basic.testing_data[i])):
                    classified += 1
            fd.write("%s,%s,%d,%d,%d\n" % (datasets[d][0], "LDAC", k, size, classified))
            
            knn = mlpy.KNN(k=3)
            knn.learn(basic.training_data, basic.training_label)
            classified = 0
            for i in range(len(basic.testing_label)):
                if (int)(basic.testing_label[i]) == (int)(knn.pred(basic.testing_data[i])):
                    classified += 1
            fd.write("%s,%s,%d,%d,%d\n" % (datasets[d][0], "KNN", k, size, classified))
            
            tree = mlpy.ClassTree(minsize=10)
            tree.learn(basic.training_data, basic.training_label)
            classified = 0
            for i in range(len(basic.testing_label)):
                if (int)(basic.testing_label[i]) == (int)(tree.pred(basic.testing_data[i])):
                    classified += 1
            fd.write("%s,%s,%d,%d,%d\n" % (datasets[d][0], "GINI", k, size, classified))
            
            ml = mlpy.MaximumLikelihoodC()
            ml.learn(basic.training_data, basic.training_label)
            classified = 0
            for i in range(len(basic.testing_label)):
                if (int)(basic.testing_label[i]) == (int)(ml.pred(basic.testing_data[i])):
                    classified += 1
            fd.write("%s,%s,%d,%d,%d\n" % (datasets[d][0], "MACL", k, size, classified))
            
            svm = mlpy.LibLinear(solver_type='mcsvm_cs', C=0.01)
            svm.learn(basic.training_data, basic.training_label)
            classified = 0
            for i in range(len(basic.testing_label)):
                if (int)(basic.testing_label[i]) == (int)(svm.pred(basic.testing_data[i])):
                    classified += 1
            fd.write("%s,%s,%d,%d,%d\n" % (datasets[d][0], "SVM", k, size, classified))
    fd.close()    