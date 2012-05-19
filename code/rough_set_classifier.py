#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import numpy
# import pylab
import copy
import sys
import math

class RoughSetClassifier(object):
    
    def __init__(self, debug=False):
        """
        """
        self.DEBUG = debug 
        self.DO_NOT_USE = 1
        self.the_best = -999.0

    def read_data(self, filepath, label_is_last=True):
        """
        label_is_last indicates where label of the class is located.
        If it is True then it is the last column, otherwise it is the first
        """
        try:
            fd = open(filepath, 'r')
            lines = fd.readlines()
            if label_is_last:
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
                    self.training_label.append(obj[0])
                    self.training_data.append(obj[1])
            else:
                for obj in self.validation_container[i]:
                    self.testing_label.append(obj[0])
                    self.testing_data.append(obj[1])
        # for secure reason check if testing + learning is equal to data
        assert(len(self.training_data) + len(self.testing_data) == len(self.data))
        return True
    
    def initialize_genetic(self, generations, mutation_prop, crossover_prop):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.generations = generations
        self.mutation_prop = mutation_prop
        self.crossover_prop = crossover_prop
    
    def create_population(self, population_size, division):
        self.division = division
        self.population_size = population_size
        
        # calculate min, max values
        attr_length = self.get_number_of_attributes()
        self.min = [999]*attr_length
        self.max = [-999]*attr_length
        
        # find min/max ranges using training data
        for i in range(attr_length):
            for j in range(len(self.training_data)):
                
                if self.min[i] > self.training_data[j][i]:
                    self.min[i] = self.training_data[j][i]
                    
                if self.max[i] < self.training_data[j][i]:
                    self.max[i] = self.training_data[j][i]
              
        population_counter = 0
        self.population = []
        attr_length = self.get_number_of_attributes()
        while population_counter < self.population_size:
            individual = []
            for _ in range(attr_length):
                individual.append(random.randint(self.DO_NOT_USE, self.division))
            number = len(filter(lambda x: not x==self.DO_NOT_USE, individual[0:-1]))
            if number == 0:
                index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                individual[index] = random.randint(2, self.division)
            # used for storing individual fitness
            individual.append(0.0)
            self.population.append(individual)
            population_counter += 1
            
    def __train_individual(self, individual):
        """
        """
        
        mapping = {}
        for pattern_number in range(len(self.training_data)):
            pattern = self.training_data[pattern_number]
            index_table = []
            for a_n in range(self.get_number_of_attributes()):
                if not individual[a_n] == self.DO_NOT_USE:
                    if pattern[a_n] < self.min[a_n]:
                        index_pattern = 0
                    elif pattern[a_n] > self.max[a_n]:
                        index_pattern = individual[a_n] - 1
                    else:
                        index_pattern = int(individual[a_n]*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
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

    def __assign_label(self, pattern, individual, mapping):
        """
        """
        
        index_table = []
        for a_n in range(self.get_number_of_attributes()):
            if not individual[a_n] == self.DO_NOT_USE:
                if pattern[a_n] < self.min[a_n]:
                    index_pattern = 0
                elif pattern[a_n] > self.max[a_n]:
                    index_pattern = individual[a_n] - 1
                else:
                    index_pattern = int(individual[a_n]*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
                index_table.append(index_pattern)

        hash_value = ''.join('.'+str(x) for x in index_table)
        values = []
        for key in mapping:
            if hash_value in mapping[key].keys():
                values.append([mapping[key][hash_value], key])
        
        if not len(values):
            return -1
        
        max_value = max(values)
        values = filter(lambda x: x[0]>=max_value[0], values)
        if len(filter(lambda x: not x[1] == values[0][1], values)) == 0:
            return values[0][1]
        
        return -1

    def run(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation(self.generations) 
            self.generations -= 1
        print "Najlepsze rozwiazanie zaklasyfikowalo %d obiektow z %d" % (self.the_best, len(self.testing_label))

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

    def __crossover(self):
        """
        """
        size = len(self.population)
        tab = random.sample(range(size), 2)
        mother = self.population[tab[0]]
        father = self.population[tab[1]]
        child_1 = []
        child_2 = []
        attr_length = self.get_number_of_attributes()
        points = random.sample(range(attr_length), len(mother)/2)
        points = sorted(points, key=lambda x: x, reverse=True)
        for i in range(attr_length):
            if len(points) and i == points[-1]:
                child_1.append(mother[i])
                child_2.append(father[i])
                points.pop()
            else:
                child_1.append(father[i])
                child_2.append(mother[i])
        # here we will store fitness value
        child_1.append(0)
        child_2.append(0)
        
        children = [child_1, child_2]
        
        for child in children:
            number = len(filter(lambda x: not x==self.DO_NOT_USE, child[0:-1]))
            if number == 0:
                index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                child[index] = random.randint(2, self.division)
            self.extra_rules.append(child)

    def __mutation(self):
        """
        """
 
        if random.random() > self.mutation_prop:
            attr_length = self.get_number_of_attributes()
            which_rule = random.sample(range(len(self.population)), 1)[0]
            rule = self.population[which_rule]
            
            a_indexes = random.sample(range(attr_length), attr_length/2)
            # remove this attribute from composing a rule
            for index in a_indexes:
                if random.randint(0, 1):
                    rule[index] = self.DO_NOT_USE
                else:
                    rule[index] = random.randint(2, self.division)
            number = len(filter(lambda x: not x==self.DO_NOT_USE, rule[0:-1]))
            if number == 0:
                index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                rule[index] = random.randint(2, self.division-1)
                
            self.extra_rules.append(rule)
                    
    def __diversify(self):
        attr_length = self.get_number_of_attributes()
        for individual in self.population:
            if random.random() >= 0.5:
                for a_n in range(attr_length):
                    unique_ids = set(range(1, self.division)) - set([individual[a_n]])
                    individual[a_n] = random.sample(unique_ids, 1)[0]
                individual[-1] = 0.0
                number = len(filter(lambda x: not x==self.DO_NOT_USE, individual[0:-1]))
                if number == 0:
                    index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                    individual[index] = random.randint(2, self.division)

    def __create_next_generation(self, generation):
        """
        """
        if generation in [800, 600, 450, 250, 150, 50]:
            self.__diversify()

        if(len(self.population)<self.population_size):
            self.create_population()

        assert(len(self.population)>1)

        self.extra_rules = []
        self.__evaluate_population(self.population)

        _sum = reduce(lambda x, y: x+y[-1], self.population, 0.0)
        _avg = _sum / (len(self.population)*1.0)
        population = filter(lambda x: x[-1] >= _avg, self.population)
        
        if self.population_size - len(population) > 0:
            val = self.population_size - len(population)
            for _ in range(val):
                self.__crossover()
                self.__mutation()
        else:
            for _ in range(3):
                self.__crossover()
                self.__mutation()

        self.extra_rules = sorted(self.extra_rules, key=lambda x: x[-1], reverse=True)[0:self.population_size]
        self.__evaluate_population(self.extra_rules)

        new_population = []
        new_population.extend(self.population)
        if len(self.extra_rules):
            new_population.extend(self.extra_rules)

        new_population = sorted(new_population, key=lambda x: x[-1], reverse=True)
        self.population = None
        self.population = new_population[0:self.population_size]

#        # update population_dict for the next generation
#        self.population_dict = {}
#        for individual in self.population:
#            rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
#            if not rule_hash in self.population_dict.keys():
#                self.population_dict[rule_hash] = 1
#
#        the_best_population = []
#        the_best_population.extend(self.population)
#        if len(self.the_best_population):
#            the_best_population.extend(self.the_best_population)
#
#        the_best_population = []
#        the_best_population.extend(self.population)
#        if len(self.the_best_population):
#            the_best_population.extend(self.the_best_population)
#
#        population = {}
#        population_dict = {}
#        for individual in the_best_population:
#            rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
#            if not rule_hash in population_dict.keys():
#                population_dict[rule_hash] = 1
#                if not individual[-1][1] in population.keys():
#                    population[individual[-1][1]] = []
#                population[individual[-1][1]].append(individual)

    def __evaluate_population(self, rule_set):
        """
        """
        population_size = len(self.population)
        final_classification = numpy.zeros((population_size, 2))
        
        # train classifier
        mapping = []
        for individual in self.population:
            mapping.append(self.__train_individual(individual))
            
        
        for p in range(len(self.testing_data)):
            for i in range(population_size):
                if self.testing_label[p] == self.__assign_label(self.testing_data[p], self.population[i], mapping[i]):
                    final_classification[i][0] += 1
                else:
                    final_classification[i][1] += 1

        for i in range(population_size):
            number_of_attributes = len(filter(lambda x: not x==(self.DO_NOT_USE), self.population[i][0:-1]))
            assert(not number_of_attributes == 0)
            strength = 2*final_classification[i][0] - 5*final_classification[i][1]
            self.population[i][-1] = strength +  final_classification[i][0]*(1.0/number_of_attributes)
        
        res = final_classification.max(axis=0)
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = copy.deepcopy(self.population)
    
        print "Liczba obiektow do rozpoznania %d" % len(self.testing_label)
        print "Rozpoznane %d, Nierozpoznane %d " % (res[0], len(self.testing_label) - res[0])

            
    def classify(self, patterns, labels):
        print "Testujacy"
        mapping = self.__train_individual(self.division)
        counter = 0;
        for p in range(len(patterns)):
            result = int(self.__assign_label(patterns[p], self.division, mapping))
            print "Rozpoznano jako %d " % result
            if int(labels[p]) == result:
                counter += 1
        print "Z %d obiektow zrozpoznano %d" % (len(patterns), counter)

if __name__ == '__main__':
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data_new.csv')
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    
    fuzzy = RoughSetClassifier(False)
    filename = 'datasets/iris.data.txt'
    #filename = 'iris.data.txt'
    if fuzzy.read_data(filepath=filename, label_is_last=False) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=0)
    fuzzy.initialize_genetic(generations=500, mutation_prop=0.3, crossover_prop=0.9)
    fuzzy.create_population(population_size=10, division=6)
    #fuzzy.classify(fuzzy.testing_data, fuzzy.testing_label)
    fuzzy.run()
    sys.exit(0)