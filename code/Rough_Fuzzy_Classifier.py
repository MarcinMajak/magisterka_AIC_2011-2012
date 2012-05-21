#! /usr/bin/python
# -*- coding: utf-8 -*-
import sys
import random
import numpy
import copy

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

    def run_for_rough_set(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation_for_rough_set(self.generations) 
            self.generations -= 1
        print "Najlepsze rozwiazanie zaklasyfikowalo %d obiektow z %d" % (self.the_best, len(self.training_label))

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
            if random.random() >= 0.5:
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
            self.__diversify()

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
            for _ in range(3):
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
    
        print "Liczba obiektow do rozpoznania %d" % len(self.testing_label)
        print "Rozpoznane %d, Nierozpoznane %d " % (res[0], len(self.testing_label) - res[0])

# ------------------------------------------------------------------------------
    def generate_membership_functions(self, divisions=3, do_not_use_prop=0.3):
        """
        """
        
        # for each attribute we generate number of functions indicated by chromosome
        # created in the previous step called rough set
        
        self.functions_per_attribute = sum(range(2, divisions+2)) + 1
        self.do_not_use_prop = do_not_use_prop 
        
        attr_length = self.get_number_of_attributes()
        self.MF = []
        for a_n in range(0, attr_length):
            self.MF.append([])
            for i in range(2, divisions+2):
                div = float((self.max[a_n] - self.min[a_n])*1.0/(i-1))

                mi = self.min[a_n]
                self.MF[a_n].append([self.USED, mi, div])

                for j in range(1, i-1):
                    mi = float(self.min[a_n] + j*1.0*div)
                    self.MF[a_n].append([self.USED, mi, div])

                mi = self.max[a_n]
                self.MF[a_n].append([self.USED, mi, div])

            # this denotes don't care function
            self.MF[a_n].append([self.NOT_USED, 0, 0])

            assert(len(self.MF[a_n]) == self.functions_per_attribute)

    def create_population_for_fuzzy(self, population_size):
        """
        """
        rules_to_generate = len(self.training_label)/2
        self.population_size = population_size
        training_data = self.training_data[0:rules_to_generate]

        attr_length = self.get_number_of_attributes()
        rules = []
        rule_number = -1
        for pattern in training_data:
            rule_number+=1
            rules.append([])
            for a_n in range(attr_length):
                result = numpy.zeros((self.functions_per_attribute-1, 2))
                for mf_n in range(self.functions_per_attribute-1):
                    result[mf_n][0] = self.__get_MF_value(pattern[a_n], self.MF[a_n][mf_n][1], self.MF[a_n][mf_n][2], a_n)
                    result[mf_n][1] = mf_n

                den = sum(result)[0]
                if den == 0.0:
                    den = 1.0
                B_k = filter(lambda x: x[0]>0, result)
                B_k = map(lambda (x,y): [x/den,y], B_k)
                B_k = sorted(B_k, key=lambda x:x[0], reverse=True)

                index = -1
                r, s = random.random(), 0
                for num in B_k:
                    s += num[0]
                    if s >= r:
                        index = int(num[1])
                        break

                assert(not index == -1)
                # append antecedent
                rules[rule_number].append(index)
            # append consequent
            rules[rule_number].append([0, 0, 0])
            
            # for each attribute rule decide if it is used or not.
            if random.random() > self.do_not_use_prop:
                numbers = random.sample(range(attr_length), int(random.uniform(0.1, 0.4)*attr_length))
                for number in numbers:
                    # the last index in the MF functions indicates don't care
                    rules[rule_number][number] = self.functions_per_attribute - 1

        # train each rule to obtain reliable results
        self.population = []
        self.population_dict = {}
        for individual in rules:
            result = self.__train_individual(individual, self.training_data, self.training_label)
            if not result[-1][0] == self.NOT_USED:
                # calculate hash value for this individual to find out if it is 
                # present in the population
                rule_hash = ''.join('.'+str(x) for x in result[0:-1])
                if not rule_hash in self.population_dict.keys():
                    self.population_dict[rule_hash] = 1
                    self.population.append(result)
        self.population = self.population[0:self.population_size]
        assert(not len(self.population) == 0)

    def __create_additional_rules(self):
        """
        """
        attr_length = self.get_number_of_attributes()
        rules = []
        rule_number = -1
        for pattern in self.data_to_train:
            rule_number+=1
            rules.append([])
            for a_n in range(attr_length):
                result = numpy.zeros((self.functions_per_attribute-1, 2))
                for mf_n in range(self.functions_per_attribute-1):
                    result[mf_n][0] = self.__get_MF_value(pattern[a_n], self.MF[a_n][mf_n][1], self.MF[a_n][mf_n][2], a_n)
                    result[mf_n][1] = mf_n

                den = sum(result)[0]
                if den == 0.0:
                    den = 1.0
                B_k = filter(lambda x: x[0]>0, result)
                B_k = map(lambda (x,y): [x/den,y], B_k)
                B_k = sorted(B_k, key=lambda x:x[0], reverse=True)

                index = -1
                r, s = random.random(), 0
                for num in B_k:
                    s += num[0]
                    if s >= r:
                        index = int(num[1])
                        break

                assert(not index == -1)
                # append antecedent
                rules[rule_number].append(index)
            # append consequent
            rules[rule_number].append([0, 0, 0])
            
            # for each attribute rule decide if it is used or not.
            if random.random() > self.do_not_use_prop:
                numbers = random.sample(range(attr_length), int(random.uniform(0.1, 0.4)*attr_length))
                for number in numbers:
                    # the last index in the MF functions indicates don't care
                    rules[rule_number][number] = self.functions_per_attribute - 1

        # train each rule to obtain reliable results
        new_rules = []
        for individual in rules:
            rule = self.__train_individual(individual, self.training_data, self.training_label)
            if not rule[-1][0] == self.NOT_USED:
                rule_hash = ''.join('.'+str(x) for x in rule[0:-1])
                if not rule_hash in self.population_dict.keys():
                    self.population_dict[rule_hash] = 1
                    new_rules.append(rule)
        if len(new_rules):
            self.extra_rules.extend(new_rules)

        self.data_to_train = []
        self.label_to_train = []


    def run_for_fuzzy(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation_for_fuzzy(self.generations) 
            self.generations -= 1
        self.__print_summary()
        print "Klasyfikacja zbioru uczacego"
        print "Klasyfikacja za pomoca najlepszego"
        self.__evaluate_population(self.training_data, self.training_label, self.the_classification)
        print "Klasyfikacja za pomoca najlepszych regul"
        self.__evaluate_population(self.training_data, self.training_label, self.the_best_population)

        print "\nKlasyfikacja zbioru testujacego"
        print "Klasyfikacja za pomoca najlepszego"
        self.__evaluate_population(self.testing_data, self.testing_label, self.the_classification)
        print "Klasyfikacja za pomoca najlepszych regul"
        self.__evaluate_population(self.testing_data, self.testing_label, self.the_best_population)

    def __is_in_MF_range(self, value, a, b, which):
        _min = self.min[which]
        _max = self.max[which]
        
        if value <= _min and a == min:
            return True

        if value >= _max and a == max:
            return True

        if value > (a-b) and value < (a+b):
            return True
        return False

    def __get_MF_value(self, value, a, b, which):
        _min = self.min[which]
        _max = self.max[which]
        
        if value <= _min and a == min:
            return 1
        
        if value >= _max and a == max:
            return 1
        
        if value >= a and value <= (a+b):
            val = round(-1.0/b*value + 1.0/b*(a+b),3)
            assert(val <= 1.0)
            return val
        
        if value >= (a-b) and value < a:
            val = round(1.0/b*value - 1.0/b*(a-b),3)
            assert(val <= 1.0)
            return val
        return 0

    def __crossover_for_fuzzy(self, rule_set, _avg):
        """
        """
        size = len(rule_set)
        tab = random.sample(range(size), 2)
        mother = rule_set[tab[0]]
        father = rule_set[tab[1]]
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
        child_1.append([0, 0, 0])
        child_2.append([0, 0, 0])
        
        children = [child_1, child_2]
        
        for child in children:
            number = len(filter(lambda x: not x==(self.functions_per_attribute - 1), child[0:-1]))
            if number == 0:
                index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                child[index] = random.randint(0, self.functions_per_attribute-2)

            new_child = self.__train_individual(child, self.training_data, self.training_label)
            if not new_child[-1][0] == self.NOT_USED:
                rule_hash = ''.join('.'+str(x) for x in new_child[0:-1])
                if not rule_hash in self.population_dict.keys():
                    self.population_dict[rule_hash] = 1
                    self.extra_rules.append(new_child)

    def __mutation_for_fuzzy(self, rule_set):
        """
        """
 
        if random.random() > self.mutation_prop:
            which_rule = random.sample(range(len(rule_set)), 1)[0]
            rule = rule_set[which_rule]
            new_rule = []
            for i in range(self.get_number_of_attributes()):
                new_rule.append(rule[i])
            new_rule.append([0, 0, 0])

            a_indexes = random.sample(range(self.get_number_of_attributes()), 1)
            # remove this attribute from composing a rule
            for index in a_indexes:
                if random.randint(0, 1):
                    new_rule[index] = self.functions_per_attribute - 1
                else:
                    new_rule[index] = random.randint(0, self.functions_per_attribute-2)
            number = len(filter(lambda x: not x==(self.functions_per_attribute - 1), new_rule[0:-1]))
            if number == 0:
                index = random.sample(range(self.get_number_of_attributes()), 1)[0]
                new_rule[index] = random.randint(0, self.functions_per_attribute-2)
            individual = self.__train_individual(new_rule, self.training_data, self.training_label)

            if not individual[-1][0] == self.NOT_USED:
                rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
                if not rule_hash in self.population_dict.keys():
                    self.population_dict[rule_hash] = 1
                    self.extra_rules.append(individual)
                    
    def __diversify_for_fuzzy(self):
        attr_length = self.get_number_of_attributes()
        for individual in self.population:
            if random.random() >= 0.5:
                print individual
                new_individual = []
                for a_n in range(attr_length):
                    unique_ids = set(range(0, self.functions_per_attribute-1)) - set([individual[a_n]])
                    new_individual.append(random.sample(unique_ids, 1)[0])
                new_individual.append([0, 0, 0])
                new_individual = self.__train_individual(new_individual, self.training_data, self.training_label)
                if new_individual[-1][0] == self.USED:
                    new_rule_hash = ''.join('.'+str(x) for x in new_individual[0:-1])
                    rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
                    del self.population_dict[rule_hash]
                    self.population_dict[new_rule_hash] = 1
                    for j in range(attr_length):
                        individual[j] = new_individual[j]
                    individual[-1][0] = new_individual[-1][0]
                    individual[-1][1] = new_individual[-1][1]
                    individual[-1][2] = new_individual[-1][2]
                print individual

    def __create_next_generation_for_fuzzy(self, generation):
        """
        """
        if generation in [800, 600, 450, 250, 150, 50]:
            attr_length = self.get_number_of_attributes()
            for rule in self.population:
                print "[",
                for i in range(attr_length):
                    print "%d " % rule[i],
                print "(%.2f %d, %.2f)" % (rule[-1][0], rule[-1][1], rule[-1][2]),
                print "]"
            self.__diversify_for_fuzzy()()

        if(len(self.population)<self.population_size):
            self.create_population_for_fuzzy(self.population_size)

        assert(len(self.population)>1)

        self.extra_rules = []
        self.data_to_train = []
        self.label_to_train = []
        self.__evaluate_population_for_fuzzy(self.training_data, self.training_label, self.population)
        self.__create_additional_rules()

        for _ in range(5):
            self.__crossover(self.population, 0)
            self.__mutation(self.population)

        self.extra_rules = sorted(self.extra_rules, key=lambda x: x[-1][0], reverse=True)[0:self.population_size]
        self.__evaluate_population_for_fuzzy(self.training_data, self.training_label, self.extra_rules)

        new_population = []
        new_population.extend(self.population)
        if len(self.extra_rules):
            new_population.extend(self.extra_rules)

        new_population = sorted(new_population, key=lambda x: x[-1][0], reverse=True)
        self.population = None
        self.population = new_population[0:self.population_size]

        # update population_dict for the next generation
        self.population_dict = {}
        for individual in self.population:
            rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
            if not rule_hash in self.population_dict.keys():
                self.population_dict[rule_hash] = 1

        the_best_population = []
        the_best_population.extend(self.population)
        if len(self.the_best_population):
            the_best_population.extend(self.the_best_population)

        the_best_population = []
        the_best_population.extend(self.population)
        if len(self.the_best_population):
            the_best_population.extend(self.the_best_population)

        population = {}
        population_dict = {}
        for individual in the_best_population:
            rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
            if not rule_hash in population_dict.keys():
                population_dict[rule_hash] = 1
                if not individual[-1][1] in population.keys():
                    population[individual[-1][1]] = []
                population[individual[-1][1]].append(individual)

        self.the_best_population = []
        for key in population.keys():
            _sum = reduce(lambda x, y: x+y[-1][0], population[key], 0.0)
            div = len(population[key])
            _avg = _sum/div
            population[key] = filter(lambda x: x[-1][0]>=_avg, population[key])
            if len(population[key]):
                self.the_best_population.extend(copy.deepcopy(population[key]))

    def __evaluate_population_for_fuzzy(self, patterns, labels, rule_set):
        """
        """

        attr_length = self.get_number_of_attributes()
        final_classification = numpy.zeros((len(rule_set), 3))
        activated = numpy.zeros(len(rule_set))
        
        for p in range(len(patterns)):
            rule_value = numpy.zeros((len(rule_set), 3))
            for r in range(len(rule_set)):
                antecedent_value = []
                rule = rule_set[r]
                for a_n in range(attr_length):
                    vMF = self.MF[a_n][rule[a_n]]
                    if vMF[0] == self.USED:
                        if self.__is_in_MF_range(patterns[p][a_n], vMF[1], vMF[2], a_n):
                            antecedent_value.append(self.__get_MF_value(patterns[p][a_n], vMF[1], vMF[2], a_n))
                        else:
                            antecedent_value = []
                            break

                if len(antecedent_value):
                    if int(rule[-1][1]) == int(labels[p]):
                        activated[r] += rule[-1][2] 
                    else:
                        activated[r] -= rule[-1][2]
                    rule_value[r] = [reduce(lambda x,y: x*y, antecedent_value, 1.0)*rule[-1][2], rule[-1][1], r]

            for i in range(len(final_classification)):
                if not rule_value[i][0] == 0.0:
                    if int(rule_value[i][1]) == labels[p]:
                        # this rule correctly recognize objects
                        final_classification[i][2] += 2*rule_value[i][0]
                    else:
                        final_classification[i][2] -= 5*rule_value[i][0]

            max_alpha = rule_value.max(axis=0)
            numbers = filter(lambda x: x[0]>=max_alpha[0], rule_value)
            if len(numbers) and len(filter(lambda x: not x[1]==numbers[0][1], numbers))==0 and not numbers[0][0] == 0:
                numbers = numbers[0]
                q = int(numbers[2])
                if int(numbers[1]) == int(labels[p]):
                    final_classification[q][0] += 1
                else:
                    final_classification[q][1] += 1
                    self.data_to_train.append(patterns[p])
                    self.label_to_train.append(labels[p])
            else:
                self.data_to_train.append(patterns[p])
                self.label_to_train.append(labels[p])

        for i in range(len(rule_set)):
            number_of_attributes = len(filter(lambda x: not x==(self.functions_per_attribute - 1), rule_set[i][0:-1]))
            assert(not number_of_attributes == 0)
            strength = 2*final_classification[i][0] - 5*final_classification[i][1] + final_classification[i][2]
            #rule_set[i][attr_length][0] = strength +  4*(1.0/number_of_attributes) + 20.0*rule_set[i][-1][2]
            rule_set[i][attr_length][0] = strength +  final_classification[i][0]*(1.0/number_of_attributes) + 20.0*rule_set[i][-1][2]
        res = final_classification.sum(axis=0)
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = copy.deepcopy(rule_set)
    
        print "Liczba obiektow do rozpoznania %d" % len(labels)
        print "Rozpoznane %d, Nierozpoznane %d " % (res[0], res[1])

    def __train_individual_for_fuzzy(self, individual, patterns, labels):
        """
        Function responsible for training individual and assigning
        strength to a given rule.
        """

        attr_length = self.get_number_of_attributes()
        beta = {}
        for i in range(len(labels)):
            rule_value = []
            for a_n in range(0, attr_length):
                vMF = self.MF[a_n][individual[a_n]]
                if vMF[0] == self.USED:
                    # check if function is in the range of a given membership function
                    if self.__is_in_MF_range(patterns[i][a_n], vMF[1], vMF[2], a_n):
                        rule_value.append(self.__get_MF_value(patterns[i][a_n], vMF[1], vMF[2], a_n))
                    else:
                        rule_value = []
                        break
            if len(rule_value):
                min_value = reduce(lambda x,y: x*y, rule_value, 1.0)
                # here we store value of min product and class label
                if not labels[i] in beta.keys():
                    beta[labels[i]] = 0.0
                beta[labels[i]] += min_value 
                    
        if len(beta.keys()) == 0:
            # this is a dummy rule.
            individual[-1][0] = self.NOT_USED
            individual[-1][1] = 0
            individual[-1][2] = 0 
        else:
            # now we have to decide which class we should assign to this rule
            # and additionally the corresponding strength.
            max_value = numpy.max(beta.values())
            _labels = filter(lambda key: beta[key]>=max_value, beta)
            if len(_labels) and len(filter(lambda x: not x==_labels[0], _labels))==0:
                label = _labels[0]

                # calculate strength factor.
                __beta_difference = 0.0
                __beta_total = 0.0
                for key in beta:
                    if not key == label:
                        __beta_difference += beta[key]
                    __beta_total += beta[key]
                if __beta_total == 0.0:
                    __beta_total = 1.0
                strength = (beta[label] - __beta_difference)/__beta_total
                if strength > 0 :
                    individual[-1][0] = self.USED
                    individual[-1][1] = label
                    individual[-1][2] = strength

                    return individual
        # this is a dummy rule because has max value 
        # for different classes
        individual[-1][0] = self.NOT_USED
        individual[-1][1] = 0
        individual[-1][2] = 0
        return individual  

    def classify(self, pattern):
        attr_length = self.get_number_of_attributes()
        ruleset_size = len(self.the_best_population)
        rule_value = numpy.zeros((ruleset_size, 3))
        for r in range(ruleset_size):
            antecedent_value = []
            rule = self.the_best_population[r]
            for a_n in range(attr_length):
                vMF = self.MF[a_n][rule[a_n]]
                if vMF[0] == self.USED:
                    if self.__is_in_MF_range(pattern[a_n], vMF[1], vMF[2], a_n):
                        antecedent_value.append(self.__get_MF_value(pattern[a_n], vMF[1], vMF[2], a_n))
                    else:
                        antecedent_value = []
                        break

            if len(antecedent_value):
                rule_value[r] = [reduce(lambda x,y: x*y, antecedent_value, 1.0)*rule[-1][2], rule[-1][1], r]

        max_alpha = rule_value.max(axis=0)
        numbers = filter(lambda x: x[0]>=max_alpha[0], rule_value)
        if len(numbers) and len(filter(lambda x: not x[1]==numbers[0][1], numbers))==0 and not numbers[0][0] == 0:
            return numbers[0][1]
        return -1
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    hybrid = RoughFuzzyClassifier()
    filename = 'datasets/pima.data.txt'
    if hybrid.read_data(filepath=filename, label_is_last=True) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    hybrid.prepare_data(k_fold_number=2)
    hybrid.k_fold_cross_validation(k=0)
    hybrid.initialize_genetic(generations=500, mutation=0.3, crossover=0.9)
    hybrid.create_population_for_rough_set(population_size=10, division=5)
    hybrid.run_for_rough_set()
    
    # here we have optimal feature space division, now apply fuzzy logic classifier
    hybrid.generate_membership_functions()
    hybrid.create_population_for_fuzzy(population_size=10)

    hybrid.run_for_fuzzy()
    size = len(hybrid.testing_label)
    classification = numpy.zeros(size)
    for i in range(size):
        if hybrid.testing_label[i] == hybrid.classify(hybrid.testing_data[i]):
            classification[i][0] = 1
    res = classification.sum()
    print "Fuzzy %d, rough %d out of %d" % (res, size)
    sys.exit(0)
    
