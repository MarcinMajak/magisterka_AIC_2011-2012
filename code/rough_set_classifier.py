#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import numpy
# import pylab
import copy
import sys

class RoughSetClassifier(object):
    
    def __init__(self, debug=False):
        """
        """
        self.DEBUG = debug 
        self.the_best = -999.0
        self.INDICATORS = {'SECURE': 1, 'POSSIBLE': 2, 'EVEN':3, 'VOID': 4}

    def read_data(self, filepath, label_location=True):
        """
        label_location indicates where label of the class is located.
        If it is True then it is the last column, otherwise it is the first
        """
        try:
            fd = open(filepath, 'r')
            lines = fd.readlines()
            if label_location:
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
    
    def initialize_genetic(self, population_size, generations, mutation_prop, crossover_prop):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_prop = mutation_prop
        self.crossover_prop = crossover_prop
    
    def divide_space(self, division, valid_attributes=None):
        self.division = division
        if not valid_attributes:
            self.valid_attributes = valid_attributes
        else:
            self.valid_attributes = self.get_number_of_attributes()
        
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
                    
        # prepare array to store factors
        array_size = [division for _ in valid_attributes]
        self.mapping = numpy.zeros(array_size)

    def run(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation(self.generations) 
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

    def __mutation(self):
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

    def __create_random_population(self):
        self.population = []
        self.population_dict = {}
        for _ in range(self.population_size):
            individual = []
            for _ in range(self.get_number_of_attributes()):
                individual.append(random.randint(0, self.functions_per_attribute-1))
            individual.append([0,0,0])
            individual = self.__train_individual(individual, self.training_data, self.training_label)
            if not individual[-1][0] == self.NOT_USED:
                rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
                if not rule_hash in self.population_dict.keys():
                    self.population_dict[rule_hash] = 1
                    self.population.append(individual)
                    
    def __diversify(self):
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

    def __create_next_generation(self, generation):
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
            self.__diversify()
            #self.__create_random_population()

        if(len(self.population)<self.population_size):
            self.create_population_for_mitchigan(number_of_rules=1)

        assert(len(self.population)>1)

        self.extra_rules = []
        self.data_to_train = []
        self.label_to_train = []
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.population)
        self.__create_additional_rules()

        for _ in range(5):
            self.__apply_crossover_for_mitchigan(self.population, 0)
            self.__apply_mutation_for_mitchigan(self.population)

        self.extra_rules = sorted(self.extra_rules, key=lambda x: x[-1][0], reverse=True)[0:self.population_size]
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.extra_rules)

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

    def __evaluate_population(self, patterns, labels, rule_set):
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

    def __print_summary(self):
        print "Liczba obiektow do rozpoznania to %d " % len(self.training_label)
        print "Najlepsze rozpoznanie to %d " % self.the_best
        print "Dla go rozpoznania funkcje wygladaly tak"
        attr_length = self.get_number_of_attributes()
        print "[",
        for rule in self.the_classification:
            print "[",
            for i in range(attr_length):
                print "%d, " % rule[i],
            print "[%.2f, %d, %.2f]" % (rule[-1][0], rule[-1][1], rule[-1][2]),
            print "]"
        print "]"
        print "Ogolnie najlepsze rozwiazanie wygladalo tak:"
        print "[",
        for rule in self.the_best_population:
            print "[",
            for i in range(attr_length):
                print "%d, " % rule[i],
            print "[%.2f, %d, %.2f]" % (rule[-1][0], rule[-1][1], rule[-1][2]),
            print "]"
        print
            

    def __train_individual(self, individual):
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

    def classify(self, rule_set):
        print "Uczacy"
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, rule_set)
        print "Testujacy"
        self.__evaluate_population_for_mitchigan(self.testing_data, self.testing_label, rule_set)

if __name__ == '__main__':
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data_new.csv')
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    
    fuzzy = RoughSetClassifier(False)
    #filename = 'datasets/wine.data.txt'
    filename = 'datasets/iris.data.txt'
    if fuzzy.read_data(filepath=filename, label_location=False) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=0)
    fuzzy.initialize_genetic(population_size=10, generations=500, mutation_prop=0.3, crossover_prop=0.9)
    fuzzy.divide_space(division=4)
    fuzzy.run()
    sys.exit(0)