#!/usr/bin/python
# -*- coding: utf-8 -*- 

import os
import random
import string
import struct
import math
import numpy
import pylab
import sys

class FuzzyLogicClassifier(object):
    def __init__(self, filepath, cross_type=5):
        """
        """
        self.DEBUG = False
        
        # used for storing data and label of class from 
        # the file.
        self.data = []
        self.label = []
        
        # used for storing data and labels for learning procedure
        self.learning_data = []
        self.learning_label = []
        # used for storing data and labels for testing procedure
        self.testing_data = []
        self.testing_data = []
        
        self.filepath = filepath
        self.cross_type = cross_type 
        
        self.USED = 1
        self.NOT_USED = 0

    def read_data(self):
        """
        """
        try:
            fd = open(self.filepath, 'r')
            lines = fd.readlines()
            self.data = [map(float, x.strip().split(',')[1:]) for x in lines]
            self.label = [map(float, x.strip().split(',')[0]) for x in lines]
            fd.close()
            self.__prepare_validation()
        except (ValueError, IOError):
            pass

    def get_number_of_attributes(self):
        """
        """
        return len(self.data[0]) if len(self.data) else 0
    
    def size_of_data(self):
        """
        """
        return len(self.data)
    
    def __prepare_validation(self):
        dict = {}
        for i in range(0, len(self.data)):
            if not self.label[i][0] in dict.keys():
                dict[self.label[i][0]] = []        
            dict[self.label[i][0]].append(self.data[i])
        
        self.validation_container = {}
        _work = True
        while _work:
            for i in range(self.cross_type):
                if not i in self.validation_container.keys():
                    self.validation_container[i] = []
                _sum = 0
                for _class in dict.keys():
                    if len(dict[_class]):
                        self.validation_container[i].append([_class, dict[_class].pop()])
                    _sum = _sum + len(dict[_class])
                if _sum == 0:
                    _work = False
                    break
            
    def cross_validate(self, round_count):
        """
        """            
        counter = self.size_of_data()/self.cross_type
        if round_count < 0 or counter == 0 or round_count*counter > self.size_of_data():
            return False
        
        self.learning_data = []
        self.learning_label = []
        self.testing_data = []
        self.testing_label = []
        
        for i in range(self.cross_type):
            if i == round_count:
                for obj in self.validation_container[round_count]:
                    self.learning_label.append(obj[0])
                    self.learning_data.append(obj[1])
            else:
                for obj in self.validation_container[i]:
                    self.testing_label.append(obj[0])
                    self.testing_data.append(obj[1])
        
        # for secure reason check if testing + learning is equal to data
        assert(len(self.learning_data) + len(self.testing_data) == len(self.data))            
        return True
    
    def save_results(self):
        """
        """
        pass
    
    # here we have functions related to genetic algorithm
    
    def __get_characteristic_points(self, mean, sigma):
        points = [-2.9 , -1.5, -1.0 , -0.5 ,0, 0.5, 1.0, 1.5, 2.9]
        x = [_x*sigma+mean for _x in points]
        y = [numpy.exp((-(_x-mean)*(_x-mean)/(2*sigma*sigma))) for _x in x ]
        
        return (x, y)
    
    def __get_function_value(self, val, mean=0, sigma=1):
        if sigma == 0:
            return 1
        return numpy.exp(-(val-mean)**2/(2*sigma**2))
    
    def intialize_genetic(self, population_size, generations, mutation, crossover, 
                          nor=8, nov=10):
        """
        """
        # initialization
        #-----------------------------------------------------------------------
        self.generations = generations
        self.mutation_prop = mutation
        self.crossover_prop = crossover
        
        # parameters connected with fuzzy-logic
        self.number_of_rules = nor
        self.number_of_variables = nov - 2
        #-----------------------------------------------------------------------
        
        # right now we want to find min, max, variance for each atrribute.
        #-----------------------------------------------------------------------
        attr_length = len(self.learning_data[0])
        self.min = [999]*attr_length
        self.max = [-999]*attr_length
        
        avg = [0]*attr_length
        variance = [0]*attr_length
        for i in range(attr_length):
            for j in range(len(self.learning_data)):
                
                if self.min[i] > self.learning_data[j][i]:
                    self.min[i] = self.learning_data[j][i]
                    
                if self.max[i] < self.learning_data[j][i]:
                    self.max[i] = self.learning_data[j][i]
                    
                avg[i] = avg[i] + self.learning_data[j][i]
                
        div = len(self.learning_data)
        avg = [float(x/div) for x in avg]                
        for i in range(attr_length):
            for j in range(len(self.learning_data)):
                val = self.learning_data[j][i] - avg[i]
                variance[i] = variance[i] + val*val
        variance = [math.sqrt(x/(div -1)) for x in variance]
        #-----------------------------------------------------------------------
        
        # now for each variable generate membership functions
        #-----------------------------------------------------------------------
        # we have n+m variables:
        # n- input variables
        # m- output variables
        self.MF = []
        for variable in range(0, attr_length):
            self.MF.append([])
            div = float((self.max[variable] - self.min[variable])*1.0/self.number_of_variables)
            for i in range(0, self.number_of_variables):
                start = self.min[variable] + i*div
                stop = start + div
                mi = random.triangular(start, stop)
                sigma = random.triangular(0.2, 5*variance[variable]) #(1.0 + random.random())*div
                assert(sigma != 0)
                self.MF[variable].append([self.USED, mi, sigma])  

            mi = self.min[variable]
            sigma = random.triangular(0.2, 5*variance[variable])
            assert(sigma != 0)
            self.MF[variable].append([self.USED, mi, sigma])
    
            mi = self.max[variable]
            sigma = random.triangular(0.2, 5*variance[variable])
            assert(sigma != 0)
            self.MF[variable].append([self.USED, mi, sigma])
            
            print "-----------------------------------------------------------------"
            print "(Function-%d, %0.3f, %0.3f)" % (variable, self.min[variable], self.max[variable])
            for mf in self.MF[variable]:
                print "Function [%0.3f, %0.3f]" % (mf[1], mf[2])
            print "-----------------------------------------------------------------"
            
            # plot functions
            self.__plot_functions(variable, self.MF[variable], self.min[variable], self.max[variable])
        
        # additionally we create membership function for output
        min_output = min(self.label)
        max_output = max(self.label)
        
        self.min.append(min_output[0])
        self.max.append(max_output[0])
        
        self.MF.append([])
        div = float((max_output[0] - min_output[0])*1.0/self.number_of_variables)
        for i in range(0, self.number_of_variables):
            start = min_output[0] + i*div
            stop = start + div
            mi = random.triangular(start, stop)
            sigma = 1.0
            assert(sigma != 0)
            self.MF[attr_length].append([self.USED, mi, sigma])  

        mi = min_output[0]
        sigma = 1.0
        assert(sigma != 0)
        self.MF[attr_length].append([self.USED, mi, sigma])
    
        mi = max_output[0]
        sigma = random.triangular(min_output[0], max_output[0])
        assert(sigma != 0)
        self.MF[attr_length].append([self.USED, mi, sigma])
        
        print "-----------------------------------------------------------------"
        print "(Function-%d, %0.3f, %0.3f)" % (attr_length, min_output[0], max_output[0])
        for mf in self.MF[attr_length]:
            print "Function [%0.3f, %0.3f]" % (mf[1], mf[2])
        print "-----------------------------------------------------------------"
        
        self.__plot_functions(attr_length, self.MF[attr_length], min_output[0], max_output[0])
        #-----------------------------------------------------------------------
        
        if self.DEBUG:
            pylab.grid(True)
            pylab.show()
            
        # generate initial population containing population_size individuals.
        #-----------------------------------------------------------------------
        # each chromosome comprises of genes. Single gene is a tuple containing
        # mean and sigma of memebership function for each variable
        
        # tab with successive numbers used for 
        # mother-father-son selection
        self.indexes = range(0,population_size)
        # array for storing fitness values for each individual
        self.fitness_index = [0]*population_size
                
        # helper array
        helper_array = [range(0, nov)]*(attr_length+1)
        
        self.populalation = []
        for i in range(0, population_size):
            # initialize new chromosome
            self.populalation.append([])
            size = nov
                
            for _ in range(self.number_of_rules):
                for variable in range(attr_length+1):
                    index = random.randint(1, size) - 1
                    val = helper_array[variable][index]
                    helper_array[variable][index] = helper_array[variable][size-1]
                    helper_array[variable][size-1] = val
                    self.populalation[i].append(self.MF[variable][val])
                size = size - 1
                
            self.fitness_index[i] = self.__evaluate_individual(self.populalation[i])
        self.number_of_variables = nov
        assert(len(self.fitness_index)==len(self.populalation))
        assert(len(self.MF)==(attr_length+1))
        for i in range(attr_length+1):
            assert(len(self.MF[i])==nov)
        print "Initialization finished\n"     
        #-----------------------------------------------------------------------
    
    def __apply_crossover(self):
        """
        """
        # select mother individual from population
        size = len(self.indexes)
        index = random.randint(1, size) - 1
        curr_val = self.indexes[index]
        self.indexes[index] = self.indexes[size - 1]
        self.indexes[size - 1] = curr_val
        mother_index = curr_val
        
        # select father individual from population
        size = size - 1
        index = random.randint(1, size) - 1
        curr_val = self.indexes[index]
        self.indexes[index] = self.indexes[size - 1]
        self.indexes[size - 1] = curr_val
        father_index = curr_val
        
        # select child individual form population
        size = size - 1
        index = random.randint(1, size) - 1
        curr_val = self.indexes[index]
        self.indexes[index] = self.indexes[size - 1]
        self.indexes[size - 1] = curr_val
        child_index = curr_val
        
        # now apply crossover
        mother = self.populalation[mother_index]
        father = self.populalation[father_index]
        child = self.populalation[child_index]
        
        for i in range(0, len(child)):
            m_mf = mother[i]
            f_mf = father[i]
            c_mf = child[i]
            if m_mf[0] == self.USED and f_mf[0] == self.USED:
                    
                child_mi = (m_mf[1] + f_mf[1])/2
                child_sigma = (m_mf[2] + f_mf[2])/2
                if child_sigma == 0:
                    child_sigma = (random.random() + 0.1)*\
                    random.randint(math.floor(self.min[i]), math.floor(self.max[i]))
            
                child[i][0] = self.USED
                child[i][1] = child_mi
                child[i][2] = child_sigma    
        self.fitness_index[child_index] = self.__evaluate_individual(child) 
        pass
    
    def __plot_functions(self, number, MF, _min, _max):
        figure = pylab.figure(number)
        
        for i in range(0, len(MF)):
            (x, y) = self.__get_characteristic_points(MF[i][1], MF[i][2])
            pylab.plot(x, y)
            
    def __apply_mutation(self):
        """
        """
        if random.random() > self.mutation_prop:
            index = random.randint(1, len(self.populalation)) - 1
            individual = self.populalation[index]
            mod = self.get_number_of_attributes() + 1
            for i in range(0, len(individual)):
                individual[i][0] = self.USED
                individual[i][1] = random.triangular(self.min[i%mod], self.max[i%mod]) 
                individual[i][2] = random.triangular(self.min[i%mod], self.max[i%mod])/self.number_of_variables
                
            self.fitness_index[index] = self.__evaluate_individual(individual)
    
    def __evaluate_individual(self, individual):
        """
        """
        attr_length = self.get_number_of_attributes()
        recognized = 0
        for i in range(len(self.learning_data)):
            rule_value = []
            for rule_number in range(0, self.number_of_rules):
                rule_value.append([])
                start = rule_number*(attr_length+1)
                stop = start + (attr_length+1)
                ruleset = individual[start:stop]
                for variable in range(0, attr_length):
                    vMF = ruleset[variable]
                    if vMF[0] == self.USED:
                        # check if function is in the range of a given 
                        # membership function
                        (x, y) = self.__get_characteristic_points(vMF[1], vMF[2])
                        if (self.learning_data[i][variable]>=x[0] and self.learning_data[i][variable]<=x[8]):
                            rule_value[rule_number].\
                            append(self.__get_function_value( \
                                    self.learning_data[i][variable], 
                                    vMF[1], 
                                    vMF[2])
                            )
                        else:
                            # stop this rule and clear previous collected values
                            rule_value[rule_number] = []
                            break
            # now we want to calculate final result for each activated rule
            nom = 0
            den = 0
            print "-----------------------------------------------------\n"
            for final_rule in range(0, len(rule_value)):
                if len(rule_value[final_rule]):
                    which_mf = final_rule*(attr_length+1) + attr_length
                    min_value = numpy.min(rule_value[final_rule])
                    den = den + min_value
                    
                    print "Iloczyn %f, funkcja na pozycji %d, a jej wartosc to %f" % (min_value, which_mf, individual[which_mf][1])
                    nom = nom + min_value*individual[which_mf][1]
            if den == 0:
                den = 1
            result = math.floor(nom/den + 0.5)
            print "Rozpoznaje jako %f, natomiast jest w label %f" % (result, self.learning_label[i])
            if result == self.learning_label[i]:
                recognized = recognized + 1
                
            print "-----------------------------------------------------\n"
             
        return recognized
        
    def run(self):
        """
        """
        while(self.generations):
            print "Running %d-th generation\n" % self.generations
            self.__apply_crossover()
            self.__apply_mutation()
            self.generations = self.generations - 1
            
    # learning and testing procedure
    def initialize_classifier(self, cross_validation):
        self.read_data()
        self.cross_validate(cross_validation)
    
    def learn_classifier(self):
        """
        """
        
        pass
    
    def test_classifier(self):
        """
        """
        self.save_results()
        pass
    
    def print_summary(self):
        print "Number of objects to recognize %d: " % len(self.learning_label)
        print "Fitness index:"
        print self.fitness_index
        
        print "Functions per individual"
        for individual in self.populalation:
            print "[",
            for gene in individual:
                print "(%.3f,%.3f)" % (gene[1], gene[2]),
            print "]\n" 
                

if __name__ == '__main__':
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data_new.csv')
    fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    if fuzzy.read_data() == False:
        print "Zły plik został podany"
        sys.exit(1)
    
    print fuzzy.cross_validate(1)
    fuzzy.intialize_genetic(50, 50, 0.3, 0.4, 8, 15)
    fuzzy.run()
    fuzzy.print_summary()
    