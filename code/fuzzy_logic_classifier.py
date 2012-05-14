#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import numpy
# import pylab
import sys

class FuzzyLogicClassifier(object):
    
    def __init__(self, debug=False):
        """
        """
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
    
    def plot_functions(self, a, b):
        x = [a-b, a, a+b]
        y = [0, 1, 0]
        return (x, y)
    
    def generate_membership_functions(self, divisions=3, do_not_use_prop=0.3):
        """
        Divide each attribute into partitions consisting of
        *divisions* number. For example if we have 4 divisions 
        we will create 15 functions per variables because of
        (2 + 3 + 4 + 5 ) plus one do not care function.
        """
        self.functions_per_attribute = sum(range(2, divisions+2)) + 1
        self.do_not_use_prop = do_not_use_prop 
        
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

    def initialize_genetic(self, population_size, generations, mutation, crossover, mitchigan):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_prop = mutation
        self.crossover_prop = crossover
        self.mitchigan_prop = mitchigan

    def create_population_for_mitchigan(self, number_of_rules):
        """
        """
        rules_to_generate = len(self.training_label)/2
        self.number_of_classes = numpy.max(self.training_label)
        self.number_of_rules = number_of_rules
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


    def run(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation(self.generations) 
            self.generations -= 1
        self.__print_summary()
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.the_best_population)

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

    def __apply_crossover_for_mitchigan(self, rule_set, _avg):
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

    def __apply_mutation_for_mitchigan(self, rule_set):
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

    def __create_next_generation(self, generation):
        """
        """
        #if (generation+1)%40 == 0:
        #    self.__create_random_population()
        
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

        population = []
        population_dict = {}
        for individual in the_best_population:
            rule_hash = ''.join('.'+str(x) for x in individual[0:-1])
            if not rule_hash in population_dict.keys():
                population_dict[rule_hash] = 1
                population.append(individual)

        # how many individual we need.
        _sum = reduce(lambda x, y: x+y[-1][0], population, 0.0)
        div = len(population)
        if div == 0:
            div = 1
        _avg = _sum*1.0/div
        self.the_best_population = None
        self.the_best_population = filter(lambda x: x[-1][0]>_avg, population)


    def __evaluate_population_for_mitchigan(self, patterns, labels, rule_set):
        """
        """

        attr_length = self.get_number_of_attributes()
        final_classification = numpy.zeros((len(rule_set), 3))
        activated = numpy.zeros(len(rule_set))
        rule_value = numpy.zeros((len(rule_set), 3))

        for p in range(len(patterns)):
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
                    activated[r] += 1
                    rule_value[r] = [reduce(lambda x,y: x*y, antecedent_value, 1.0)*rule[attr_length][2], rule[attr_length][1], r] 
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
                if int(numbers[1]) == labels[p]:
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
            #if final_classification[i][0] == 0 and final_classification[i][1] == 0:
            #    strength = -1000
            #else:
            strength = 5*final_classification[i][0] - 10*final_classification[i][1]
            rule_set[i][attr_length][0] = strength + final_classification[i][2]+ 10*activated[i] + 4*(1.0/number_of_attributes) + 10*rule_set[i][attr_length][2]
        res = final_classification.sum(axis=0)
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = rule_set
        print "Liczba obiektow do rozpoznania %d" % len(labels)
        print "Rozpoznane %d, Nierozpoznane %d " % (res[0], res[1])

    def __print_summary(self):
        print "Liczba obiektow do rozpoznania to %d " % len(self.training_label)
        print "Najlepsze rozpoznanie to %d " % self.the_best
        print "Dla najlepszego rozpoznania funkcje wygladaly tak"
        attr_length = self.get_number_of_attributes()
        for rule in self.the_classification:
            print "[",
            for i in range(attr_length):
                print "%d " % rule[i],
            print "(%.2f %d, %.2f)" % (rule[-1][0], rule[-1][1], rule[-1][2]),
            print "]"
        print "Ogolnie najlepsze rozwiazanie wygladalo tak:"
        for rule in self.the_best_population:
            print "[",
            for i in range(attr_length):
                print "%d " % rule[i],
            print "(%.2f %d, %.2f)" % (rule[-1][0], rule[-1][1], rule[-1][2]),
            print "]"
            

    def __train_individual(self, individual, patterns, labels):
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

if __name__ == '__main__':
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data_new.csv')
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    
    fuzzy = FuzzyLogicClassifier()
    #filename = 'wine.data.txt'
    filename = 'datasets/iris.data.txt'
    fuzzy = FuzzyLogicClassifier(False)
    if fuzzy.read_data(filepath=filename, label_location=False) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=1)
    fuzzy.generate_membership_functions(divisions=4)
    fuzzy.initialize_genetic(population_size=10, generations=500, mutation=0.3, crossover=0.9, mitchigan=0.5)
    fuzzy.create_population_for_mitchigan(number_of_rules=1)
    fuzzy.run()
    sys.exit(0)