#!/usr/bin/python
# -*- coding: utf-8 -*- 

import sys
from consts import *

"""
This file contains basic Rough Sets algorithm.
"""

class RoughSetsClassifier(object):
    
    def __init__(self, use_modification=False, debug=False):
        """
        """
        self.DEBUG = debug 
        self.USE_MODIFICATION = use_modification

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
    
    def train(self, granulation):
        self.granulation = granulation
        self.mapping = {}
        self.__train(self.mapping, self.granulation)
        
    def prepare_dataset(self):
        
        # calculate min, max values
        attr_length = self.get_number_of_attributes()
        self.min = [999]*attr_length
        self.max = [-999]*attr_length
        # find min/max ranges using training data
        attr_length = self.get_number_of_attributes()
        for i in range(attr_length):
            for j in range(len(self.training_data)):
                
                if self.min[i] > self.training_data[j][i]:
                    self.min[i] = self.training_data[j][i]
                    
                if self.max[i] < self.training_data[j][i]:
                    self.max[i] = self.training_data[j][i]
            
    def __train(self, mapping, granulation):
        """
        """
        if granulation < 2:
            mapping = {}
            return {}
        
        for pattern_number in range(len(self.training_data)):
            pattern = self.training_data[pattern_number]
            index_table = []
            for a_n in range(self.get_number_of_attributes()):
                if pattern[a_n] < self.min[a_n]:
                    index_pattern = 0
                elif pattern[a_n] > self.max[a_n]:
                    index_pattern = granulation - 1
                else:
                    index_pattern = int(granulation*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
                index_table.append(index_pattern)
            hash_value = ''.join('.'+str(x) for x in index_table)
            
            if not self.training_label[pattern_number] in mapping.keys():
                mapping[self.training_label[pattern_number]] = {}
            if not hash_value in mapping[self.training_label[pattern_number]].keys():
                mapping[self.training_label[pattern_number]][hash_value] = 0
            mapping[self.training_label[pattern_number]][hash_value] += 1
        counter = 0
        for key in mapping:
            if self.DEBUG:
                print key
                for hash_key in mapping[key]:
                    print "%s %d " % (hash_key, mapping[key][hash_key])
            counter += reduce(lambda x, y: x + y, mapping[key].values(), 0)
        assert(counter == len(self.training_data))
        return mapping

    def __assign_label(self, pattern, mapping, granulation):
        """
        """
        
        if granulation < 2:
            return -1
        
        index_table = []
        for a_n in range(self.get_number_of_attributes()):
            if pattern[a_n] < self.min[a_n]:
                index_pattern = 0
            elif pattern[a_n] > self.max[a_n]:
                index_pattern = granulation - 1
            else:
                index_pattern = int(granulation*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
            index_table.append(index_pattern)

        hash_value = ''.join('.'+str(x) for x in index_table)
        values = []
        for key in mapping:
            if hash_value in mapping[key].keys():
                values.append([mapping[key][hash_value], key])
        
        if not len(values):
            if self.USE_MODIFICATION:
                mapping = {}
                self.__train(mapping, granulation - 1)
                return self.__assign_label(pattern, mapping, granulation-1)
            return -1
        
        max_value = max(values)
        values = filter(lambda x: x[0]>=max_value[0], values)
        if len(filter(lambda x: not x[1] == values[0][1], values)) == 0:
            return values[0][1]
        
        if self.USE_MODIFICATION:
            mapping = {}
            self.__train(mapping, granulation - 1)
            return self.__assign_label(pattern, mapping, granulation-1)
        return -1        
        

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

    def save_results(self):
        """
        """
        return 1

    # private methods
    # --------------------------------------------------------------------------
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
                
    def classify(self, pattern):
        return self.__assign_label(pattern, self.mapping, self.granulation)

if __name__ == '__main__':
    

    USE_MODIFICATION = False
    DEBUG = False
    MAX_GRANULATION = 18
    
    result_file = 'results/rough_sets_classifier.csv'
    try:
        fd = open(result_file, 'w')
    except IOError:
        print "Wrong path for results file"
        sys.exit(1)
    
    for d in range(len(datasets)):
        for g in range(2, MAX_GRANULATION, 1):
            for k in range(K_FOLD_NUMBER):
                rough_basic = RoughSetsClassifier(use_modification=False, debug=DEBUG)
                filename = 'datasets/%s' % datasets[d][0]
                if rough_basic.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1])) == False:
                    print "Error with opening the file. Probably you have given wrong path"
                    sys.exit(1)
                rough_basic.prepare_data(k_fold_number=K_FOLD_NUMBER)
                rough_basic.k_fold_cross_validation(k=k)
                rough_basic.prepare_dataset()
                rough_basic.train(granulation=g)
                
                size = len(rough_basic.testing_label)
                
                classified = 0
                for i in range(size):
                    if rough_basic.testing_label[i] == rough_basic.classify(rough_basic.testing_data[i]):
                        classified += 1
                fd.write("%s,%s,%d,%d,%d,%d\n" % (datasets[d][0], "Basic Rough", g, k, size, classified))
                   
                rough_advanced = RoughSetsClassifier(use_modification=True, debug=DEBUG)
                if rough_advanced.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1])) == False:
                    print "Error with opening the file. Probably you have given wrong path"
                    sys.exit(1)
                rough_advanced.prepare_data(k_fold_number=K_FOLD_NUMBER)
                rough_advanced.k_fold_cross_validation(k=k)
                rough_advanced.prepare_dataset()
                rough_advanced.train(granulation=g)
                
                classified = 0
                for i in range(size):
                    if rough_advanced.testing_label[i] == rough_advanced.classify(rough_advanced.testing_data[i]):
                        classified += 1
                fd.write("%s,%s,%d,%d,%d,%d\n" % (datasets[d][0], "Advanced Rough", g, k, size, classified))
                fd.flush()
    fd.close()