#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random
import numpy
import copy
import mlpy
from consts import *

class RoughFuzzyClassifier(object):
    def __init__(self, debug=False):
        """
        """
        self.DEBUG = debug 
        self.DO_NOT_USE = 1
        self.the_best = -999.0

    def read_data(self, filepath, label_is_last=True):
        """
        params:
        *filepath*- indicates path of file with dataset
        *label_is_last*-
        indicates where label of the class is located.
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
    
    def initialize_genetic(self, generations, mutation_prop, crossover_prop):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.generations = generations
        self.mutation_prop = mutation_prop
        self.crossover_prop = crossover_prop
    
    def create_population_for_rough_set(self, population_size, division):
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
            
    def __train_individual_for_rough_set(self, individual):
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

    def __hybrid_assign_label(self, pattern, individual, mapping):
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
            membership_values = {}
            for a_n in range(self.get_number_of_attributes()):
                if not self.the_classification[a_n] == self.DO_NOT_USE:
                    
                    current_attribute_label = []
                    # teraz musimy sie dowiedziec jaki index ma ten pattern wynikajacy
                    # z podzialu wykorzystujac odpowiednia granulacje.
                    if pattern[a_n] < self.min[a_n]:
                        index_pattern = 0
                    elif pattern[a_n] > self.max[a_n]:
                        index_pattern = individual[a_n] - 1
                    else:
                        index_pattern = int(individual[a_n]*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
                        
                    # hack here
                    index_pattern = 0
                    # wyliczamy wartosc membeship_function dla danej klasy
                    if index_pattern in self.MF[a_n].keys():
                        for label in self.MF[a_n][index_pattern]:
                            val = self.__get_MF_value(
                                                      pattern[a_n], 
                                                      self.MF[a_n][index_pattern][label][0], 
                                                      self.MF[a_n][index_pattern][label][1], 
                                                      self.MF[a_n][index_pattern][label][2], 
                                                      a_n
                                                      )
                            current_attribute_label.append([val, label])
                    else:
                        current_attribute_label.append([0, -1])
                    
                    for tab in current_attribute_label:
                        if not tab[1] in membership_values.keys():
                            membership_values[tab[1]] = 0
                        membership_values[tab[1]] += tab[0]
                    # max_label = max(current_attribute_label)
                    # l = filter(lambda x: x[0] >= max_label[0], current_attribute_label)
                    # if not l[0][1] in membership_values.keys():
                    #    membership_values[l[0][1]] = 0
                    
                    # membership_values[l[0][1]] += l[0][0]
            if len(membership_values.keys()):
                max_label_value = max(membership_values.values())
                val = filter(lambda x: membership_values[x] >= max_label_value, membership_values)
                if len(val) == 1:
                    return val[0]
            
            return -1
        
        max_value = max(values)
        values = filter(lambda x: x[0]>=max_value[0], values)
        #if len(filter(lambda x: not x[1] == values[0][1], values)) == 0:
        #    return values[0][1]
        if len(values) == 1:
            return values[0][1]
        
        # teraz mamy do czynienia z regula niepewna. Chcemy sprawdzic do jakiej
        # klasy moze najprawdopodobniej nalezec
        membership_values = {}
        for a_n in range(self.get_number_of_attributes()):
            if not self.the_classification[a_n] == self.DO_NOT_USE:
                
                current_attribute_label = []
                # teraz musimy sie dowiedziec jaki index ma ten pattern wynikajacy
                # z podzialu wykorzystujac odpowiednia granulacje.
                if pattern[a_n] < self.min[a_n]:
                    index_pattern = 0
                elif pattern[a_n] > self.max[a_n]:
                    index_pattern = individual[a_n] - 1
                else:
                    index_pattern = int(individual[a_n]*(pattern[a_n] - self.min[a_n])/(self.max[a_n]-self.min[a_n]))
                # hack here
                index_pattern = 0
                # wyliczamy wartosc membeship_function dla danej klasy
                if index_pattern in self.MF[a_n].keys():
                    for label in self.MF[a_n][index_pattern]:
                        val = self.__get_MF_value(
                                                  pattern[a_n], 
                                                  self.MF[a_n][index_pattern][label][0], 
                                                  self.MF[a_n][index_pattern][label][1], 
                                                  self.MF[a_n][index_pattern][label][2], 
                                                  a_n
                                                  )
                        current_attribute_label.append([val, label])
                else:
                    current_attribute_label.append([0, -1])
                
                for tab in current_attribute_label:
                    if not tab[1] in membership_values.keys():
                        membership_values[tab[1]] = 0
                    membership_values[tab[1]] += tab[0]
                # max_label = max(current_attribute_label)
                # l = filter(lambda x: x[0] >= max_label[0], current_attribute_label)
                # if not l[0][1] in membership_values.keys():
                #    membership_values[l[0][1]] = 0
                
                # membership_values[l[0][1]] += l[0][0]
                                    
                # szukamy najwiekszej wartosci wsrod wspolczynnikow
                # max_label = max(current_attribute_label)
                # l = filter(lambda x: x[0] >= max_label[0], current_attribute_label)
                # if not l[0][1] in membership_values.keys():
                #    membership_values[l[0][1]] = 0
                
                # membership_values[l[0][1]] += l[0][0]
        if len(membership_values.keys()):
            max_label_value = max(membership_values.values())
            val = filter(lambda x: membership_values[x] >= max_label_value, membership_values)
            if len(val) == 1:
                return val[0]
        return -1

    def run_for_rough_set(self):
        while self.generations:
            #print "Generation %d" % self.generations
            self.__create_next_generation_for_rough_set(self.generations) 
            self.generations -= 1
        #print "Najlepsze rozwiazanie zaklasyfikowalo %d obiektow z %d" % (self.the_best, len(self.testing_label))
        #print "Najlepsze rozwiazanie wygladalo tak"
        #for value in self.the_classification:
        #    if value == self.DO_NOT_USE:
        #        print "NOT_USED",
        #    else:
        #        print value,

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

    def __crossover_for_rough_set(self):
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

    def __mutation_for_rough_set(self):
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
                    
    def __diversify_for_rough_set(self):
        attr_length = self.get_number_of_attributes()
        for individual in self.population:
            if random.random() >= 0.3:
                for a_n in range(attr_length):
                    unique_ids = set(range(1, self.division)) - set([individual[a_n]])
                    individual[a_n] = random.sample(unique_ids, 1)[0]
                individual[-1] = 0.0
                number = len(filter(lambda x: not x==self.DO_NOT_USE, individual[0:-1]))
                if number == 0:
                    index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                    individual[index] = random.randint(2, self.division)

    def __create_next_generation_for_rough_set(self, generation):
        """
        """
        if generation in [800, 600, 450, 250, 150, 50]:
            self.__diversify_for_rough_set()

        if(len(self.population)<self.population_size):
            self.create_population_for_rough_set(self.population_size, self.division)

        assert(len(self.population)>1)

        self.extra_rules = []
        self.__evaluate_population_for_rough_set(self.population)

        _sum = reduce(lambda x, y: x+y[-1], self.population, 0.0)
        _avg = _sum / (len(self.population)*1.0)
        population = filter(lambda x: x[-1] >= _avg, self.population)
        
        if self.population_size - len(population) > 0:
            val = self.population_size - len(population)
            for _ in range(val):
                self.__crossover_for_rough_set()
                self.__mutation_for_rough_set()
        else:
            for _ in range(5):
                self.__crossover_for_rough_set()
                self.__mutation_for_rough_set()

        self.extra_rules = sorted(self.extra_rules, key=lambda x: x[-1], reverse=True)[0:self.population_size]
        self.__evaluate_population_for_rough_set(self.extra_rules)

        new_population = []
        new_population.extend(self.population)
        if len(self.extra_rules):
            new_population.extend(self.extra_rules)

        new_population = sorted(new_population, key=lambda x: x[-1], reverse=True)
        self.population = None
        self.population = new_population[0:self.population_size]

    def __evaluate_population_for_rough_set(self, rule_set):
        """
        """
        population_size = len(self.population)
        final_classification = numpy.zeros((population_size, 2))
        
        # train classifier
        mapping = []
        for individual in self.population:
            mapping.append(self.__train_individual_for_rough_set(individual))
            
        
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
        
        indexes = final_classification.argmax(axis=0)
        res = final_classification[indexes[0]]
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = copy.deepcopy(self.population[indexes[0]])
            self.the_mapping = copy.deepcopy(mapping[indexes[0]])
    
        #print "Liczba obiektow do rozpoznania %d" % len(self.testing_label)
        #print "Rozpoznane %d, Nierozpoznane %d " % (res[0], len(self.testing_label) - res[0])
        

    def calculate_histogram(self):
        """
        calculate histogram for each attribute which is used in 
        the best chromosome.
        """
        
        rule = self.the_classification
        histogram_per_attribute = []
        for i in range(self.get_number_of_attributes()):
            histogram_per_attribute.append({})
            if not rule[i] == self.DO_NOT_USE:
                # for this attribute we want to calculate histogram using training data
                for p in range(len(self.training_data)):
                    # find out about index of this attribute
                    index = 0#int(rule[i]*(self.training_data[p][i] - self.min[i])/(self.max[i] - self.min[i]))
                    if not index in histogram_per_attribute[i].keys():
                        histogram_per_attribute[i][index] = {}
                    if not self.training_label[p] in histogram_per_attribute[i][index].keys():
                        histogram_per_attribute[i][index][self.training_label[p]] = []
                    
                    histogram_per_attribute[i][index][self.training_label[p]].append(self.training_data[p][i])
        mf = []
        for i in range(self.get_number_of_attributes()):
            mf.append({})
            if not rule[i] == self.DO_NOT_USE:
                division_histogram = histogram_per_attribute[i]
                for div in division_histogram.keys():
                    if not div in mf[i].keys():
                        mf[i][div] = {}
                    for label in division_histogram[div].keys():                            
                        data = division_histogram[div][label]
                        hist, bin_edges = numpy.histogram(data, bins=10)
                        d = bin_edges[1] - bin_edges[0]
                        # a = reduce(lambda x, y: x+y, [(bin_edges[j]+d/2)*d*hist[j] for j in range(len(hist))], 0.0)  /(d*hist.sum())
                        # hack here
                        a = bin_edges[hist.argmax()]
                        b_left = bin_edges[0]
                        b_right = bin_edges[len(bin_edges) - 1]
                        mf[i][div][label] = [a, abs(a - b_left), abs(a - b_right)]
                        
        for i in range(self.get_number_of_attributes()):
            if not rule[i] == self.DO_NOT_USE:
                if len(mf[i].keys()):
                    for g_key in mf[i].keys():
                        div = (self.max[i] - self.min[i])/rule[i]
                        _min = self.min[i] + g_key*div
                        _max = _min + div
                        # hack here
                        _min = self.min[i]
                        _max = self.max[i]
                        
                        _mf = mf[i][g_key].values()
                        _mf = sorted(_mf, key=lambda x: x[0])
                        for m in range(len(_mf)):
                            if m == 0:
                                _mf[0][1] = abs(_min - _mf[0][0]) 
                            else:
                                if (_mf[m][0] - _mf[m][1]) > _mf[m-1][0]:
                                    _mf[m][1] = _mf[m][0] - _mf[m-1][0]
                            if m == (len(_mf) - 1):
                                _mf[m][2] = abs(_max - _mf[m][0])
                            else:
                                if _mf[m][0] + _mf[m][2] < _mf[m+1][0]:
                                    _mf[m][2] = _mf[m+1][0] - _mf[m][0]
        self.MF = mf

    def __is_in_MF_range(self, value, a, b_left, b_right, which):
        _min = self.min[which]
        _max = self.max[which]
        
        if value <= _min and a == min:
            return True

        if value >= _max and a == max:
            return True

        if value > (a-b_left) and value < (a+b_right):
            return True
        return False

    def __get_MF_value(self, value, a, b_left, b_right, which):
        _min = self.min[which]
        _max = self.max[which]
        
        if value <= _min and a == min:
            return 1
        
        if value >= _max and a == max:
            return 1
        
        if value >= a and value <= (a+b_right):
            val = round(-1.0/b_right*value + 1.0/b_right*(a+b_right),3)
            assert(val <= 1.0)
            return val
        
        if value >= (a-b_left) and value < a:
            val = round(1.0/b_left*value - 1.0/b_left*(a-b_left),3)
            assert(val <= 1.0)
            return val
        return 0


    def classify(self, pattern):
        return self.__hybrid_assign_label(pattern, self.the_classification, self.the_mapping)
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    
    DEBUG = False
    result_file = 'results/rough_fuzzy_classifier.csv'
    try:
        fd = open(result_file, 'w')
    except IOError:
        print "Wrong path for results file"
        sys.exit(1)    
    
    for d in range(len(datasets)):
        for k in range(K_FOLD_NUMBER):
            for r in range(3):    
                print "STARTED iteration %d with %d-fold for %s dataset" % (r, k, datasets[d][0])
                hybrid = RoughFuzzyClassifier()
                filename = 'datasets/%s' % datasets[d][0]
                if hybrid.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1])) == False:
                    print "Error with opening the file. Probably you have given wrong path"
                    sys.exit(1)
                hybrid.prepare_data(k_fold_number=K_FOLD_NUMBER)
                hybrid.k_fold_cross_validation(k=k)
                hybrid.initialize_genetic(generations=300, mutation_prop=MUTATION_PROP, crossover_prop=CROSS_OVER_PROP)
                hybrid.create_population_for_rough_set(population_size=POPULATION_SIZE, division=MAX_GRANULATION)
            
                hybrid.run_for_rough_set()
                hybrid.calculate_histogram()
                
                size = len(hybrid.testing_label)
                classified = 0
                for i in range(size):
                    if hybrid.testing_label[i] == hybrid.classify(hybrid.testing_data[i]):
                        classified += 1
                        
                # calculate how many attributes are used in the best classification
                active_attributes = 0
                for a in range(hybrid.get_number_of_attributes()):
                    if not hybrid.the_classification[a] == hybrid.DO_NOT_USE:
                        active_attributes += 1
                print "FINISHED iteration %d with %d-fold for %s dataset" % (r, k, datasets[d][0]) 
                print "It managed to classify %d out of %d" % (classified, size)
                fd.write("%s,%d,%d,%d,%d,%d,%d\n" % (datasets[d][0], r, k, size, classified, hybrid.get_number_of_attributes(), active_attributes))
                fd.flush()
    fd.close()                                                                  