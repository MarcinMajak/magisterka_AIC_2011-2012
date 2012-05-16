#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import numpy
import copy
import sys
import re

class FuzzyLogicClassifier(object):
    
    def __init__(self, debug=False):
        """
        """
        self.LABEL_POSITION = -2
        
        self.DEBUG = debug 
        self.USED = 1
        self.NOT_USED = 0
        self.the_best = -999.0
        self.the_best_rule_set = []
        self.extra_rules = []
        self.data_to_train = []
        self.label_to_train = []
        self.the_best_population = []
        self.the_classification = []

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

    def generate_population(self, population_size=20, functions=3, possible_divisions=5):
        """
        Responsible for generating membership functions 
        """
        self.functions_per_attribute = functions
        self.possible_divisions = possible_divisions
        self.population_size = population_size

        attr_length = self.get_number_of_attributes()
        self.min = [999]*attr_length
        self.max = [-999]*attr_length

        for i in range(attr_length):
            for j in range(len(self.training_data)):
                
                if self.min[i] > self.training_data[j][i]:
                    self.min[i] = self.training_data[j][i]
                    
                if self.max[i] < self.training_data[j][i]:
                    self.max[i] = self.training_data[j][i]

        self.population = []
        attr_length = self.get_number_of_attributes()
        population_counter = 0
        while population_counter < self.population:
            individual = []
            for _ in range(attr_length):
                tab = [1 if random.random() >=0.5 else 0 for _ in range(3)]
                if sum(tab) == 0:
                    tab[random.randint(1, self.functions_per_attribute)-1] = 1
                individual.append(''.join([str(x) for x in tab]))

            for _ in range(attr_length):
                individual.append(random.randint(1, possible_divisions-1))

            # for storing class
            individual.append(0)
            # for storing fitness value
            individual.append(0.0)
            self.__train_individual(individual)
            
            if not individual[self.LABEL_POSITION] == 0:
                population_counter += 1
                self.population.append(individual)

    def initialize_genetic(self, generations, mutation_prop, crossover_prop):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.generations = generations
        self.mutation_prop = mutation_prop
        self.crossover_prop = crossover_prop

    def __create_additional_rules(self):
        """
        """
        attr_length = self.get_number_of_attributes()
        rules = []
        rule_number = -1
        for pattern in self.data_to_train:
            rule_number+=1
            rules.append([0 for _ in range(2*attr_length+2)])
            division = random.randint(1, self.functions_per_attribute-1)
            for a_n in range(attr_length):
                result = numpy.zeros((self.functions_per_attribute, 3))
                for mf_n in range(self.functions_per_attribute):
                    tab = [0 for _ in range(self.functions_per_attribute)]
                    tab[a_n] = 1
                    val = ''.join(tab)
                    result[mf_n][0] = self.__decode(pattern[a_n], val, division, a_n)
                    result[mf_n][1] = division
                    result[mf_n][2] = val

                den = sum(result)[0]
                if den == 0.0:
                    den = 1.0
                B_k = filter(lambda x: x[0]>0, result)
                B_k = map(lambda (x,y): [x/den,y], B_k)
                B_k = sorted(B_k, key=lambda x:x[0], reverse=True)

                index = -1
                div = 1
                r, s = random.random(), 0
                for num in B_k:
                    s += num[0]
                    if s >= r:
                        div = num[1]
                        index = num[2]
                        break

                assert(not index == -1)
                # append antecedent
                rules[rule_number][a_n] = index
                rules[rule_number][a_n + attr_length] = div  
            
            # for each attribute rule decide if it is used or not.
            if random.random() > self.do_not_use_prop:
                numbers = random.sample(range(attr_length), int(random.uniform(0.1, 0.4)*attr_length))
                for number in numbers:
                    # the last index in the MF functions indicates don't care
                    rules[rule_number][number] = ''.join(['1' for _ in range(self.functions_per_attribute)])

        # train each rule to obtain reliable results
        new_rules = []
        for individual in rules:
            rule = self.__train_individual(individual)
            if not rule[self.LABEL_POSITION] == 0:
                new_rules.append(rule)
        if len(new_rules):
            self.extra_rules.extend(new_rules)

        self.data_to_train = []
        self.label_to_train = []

    def run(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation(self.generations) 
            self.generations -= 1
        self.__print_summary()
        print "Klasyfikacja zbioru uczacego"
        print "Klasyfikacja za pomoca najlepszego"
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.the_classification)
        print "Klasyfikacja za pomoca najlepszych regul"
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.the_best_population)

        print "\nKlasyfikacja zbioru testujacego"
        print "Klasyfikacja za pomoca najlepszego"
        self.__evaluate_population_for_mitchigan(self.testing_data, self.testing_label, self.the_classification)
        print "Klasyfikacja za pomoca najlepszych regul"
        self.__evaluate_population_for_mitchigan(self.testing_data, self.testing_label, self.the_best_population)

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
    
    def __train_individual(self, individual):
        """
        Function responsible for training individual and assigning
        strength to a given rule.
        """
        
        label_dict = {}
        attr_length = self.get_number_of_attributes()
        for pattern_number in range(len(self.training_label)):
            rule_value = []
            for a_n in range(attr_length):
                function_code = individual[a_n]
                division = individual[attr_length + a_n]
                rule_value.append(self.__decode(self.training_data[pattern_number][a_n] ,function_code, division, a_n))

            value = reduce(lambda x,y: x*y, rule_value, 1.0)
            if value:
                if not self.training_label[pattern_number] in label_dict.keys():
                    label_dict[self.training_label[pattern_number]] = 0.0
                label_dict[self.training_label[pattern_number]] += value
        if len(label_dict.keys()) == 0:
            individual[self.LABEL_POSITION] = 0
            return
        max_value = numpy.max(label_dict.values())
        _labels = filter(lambda key: label_dict[key]>=max_value, label_dict)
        if len(_labels) and len(filter(lambda x: not x==_labels[0], _labels))==0:
            individual[self.LABEL_POSITION] = _labels[0]
            return
        individual[self.LABEL_POSITION] = 0
    
    def __decode(self, value, function_code, division, a_n):
        """
        """
        
        if re.match("[1]{3}", function_code):
            return 1
        val = 0
        div = (self.max[a_n] - self.min[a_n])*1.0/self.possible_divisions
        for i in range(self.functions_per_attribute):
            a = [self.min[a_n], self.min[a_n] + division*div, self.max[a_n]]
            b_left = [1, a[1]-self.min[a_n], self.max[a_n] - a[1]]
            b_right = [a[1] - self.min[1], a[1] - self.min[a_n], division*div] 
            if function_code[i] == '1':
                val += self.__get_MF_value(value, a[i], b_left[i], b_right[i], a_n)
        return val    

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

    def __crossover(self):
        """
        """
        size = len(self.population)
        if size < 3:
            return
        
        tab = random.sample(range(size), 2)
        mother = self.population[tab[0]]
        father = self.population[tab[1]]
        
        child_1 = []
        child_2 = []
        
        attr_length = self.get_number_of_attributes()
        points = random.sample(range(2*attr_length), (len(mother)-2)/2)
        points = sorted(points, key=lambda x: x, reverse=True)
        for i in range(2*attr_length):
            if len(points) and i == points[-1]:
                child_1.append(mother[i])
                child_2.append(father[i])
                points.pop()
            else:
                child_1.append(father[i])
                child_2.append(mother[i])
        # class
        child_1.append(0)
        # fitness value
        child_1.append(0.0)
        
        # class
        child_2.append(0)
        # fitness value
        child_1.append(0.0)
        
        children = [child_1, child_2]
        
        for child in children:
            new_child = self.__train_individual(child)
            if not new_child[self.LABEL_POSITION] == 0:
                    self.extra_rules.append(new_child)

    def __mutation(self):
        """
        """
 
        if random.random() > self.mutation_prop:
            which_rule = random.sample(range(len(self.population)), 1)[0]
            rule = self.population[which_rule]
            new_rule = []
            for i in range(2*self.get_number_of_attributes()):
                new_rule.append(rule[i])
            # class
            new_rule.append(0)
            # fitness
            new_rule.append(0.0)

            a_indexes = random.sample(range(2*self.get_number_of_attributes()), 2)
            # remove this attribute from composing a rule
            for index in a_indexes:
                if index > self.get_number_of_attributes():
                    new_rule[index] = random.randint(1, self.possible_divisions - 1)
                else:
                    pos = random.randint(0, self.functions_per_attribute)
                    tab = list(new_rule[index])
                    tab[pos] = (tab[pos] + 1)%2
                    new_rule[index] = ''.join(tab)
                    
            individual = self.__train_individual(new_rule)
            if not individual[self.LABEL_POSITION] == 0:
                self.extra_rules.append(individual)

    def __diversify(self):
        
    
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
        self.__evaluate_population(self.training_data, self.training_label, self.population)
        self.__create_additional_rules()

        for _ in range(5):
            self.__crossover()
            self.__mutation()

        self.extra_rules = sorted(self.extra_rules, key=lambda x: x[-1], reverse=True)[0:self.population_size]
        self.__evaluate_population(self.training_data, self.training_label, self.extra_rules)

        new_population = []
        new_population.extend(self.population)
        if len(self.extra_rules):
            new_population.extend(self.extra_rules)

        new_population = sorted(new_population, key=lambda x: x[-1], reverse=True)
        self.population = None
        self.population = new_population[0:self.population_size]

    def __evaluate_population(self, patterns, labels, rule_set):
        """
        """

        attr_length = self.get_number_of_attributes()
        final_classification = numpy.zeros((self.population_size, 3))

        for p in range(len(patterns)):
            rule_value = numpy.zeros((self.population_size, 3))
            for r in range(self.population_size):
                antecedent_value = []
                rule = rule_set[r]
                for a_n in range(attr_length):
                    function_code = rule[a_n]
                    division = rule[a_n + attr_length]
                    antecedent_value.append(self.__decode(patterns[p][a_n], function_code, division, a_n))

                rule_value[r] = [reduce(lambda x,y: x*y, antecedent_value, 1.0), rule[self.LABEL_POSITION], r]


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

        for i in range(self.population_size):
            rule_set[-1] = 2*final_classification[i][0] - 5*final_classification[i][1]
        res = final_classification.sum(axis=0)
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = copy.deepcopy(self.population)
    
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

    def classify(self, rule_set):
        print "Uczacy"
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, rule_set)
        print "Testujacy"
        self.__evaluate_population_for_mitchigan(self.testing_data, self.testing_label, rule_set)

if __name__ == '__main__':
    fuzzy = FuzzyLogicClassifier()
    filename = 'datasets/wine.data.txt'
    #filename = 'iris.data.txt'
    fuzzy = FuzzyLogicClassifier(False)
    if fuzzy.read_data(filepath=filename, label_location=True) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=0)
    fuzzy.generate_population(population_size=20, functions=3, possible_divisions=5)
    fuzzy.initialize_genetic(population_size=10, generations=500, mutation=0.3, crossover=0.9, mitchigan=0.5)
    fuzzy.run()
    sys.exit(0)