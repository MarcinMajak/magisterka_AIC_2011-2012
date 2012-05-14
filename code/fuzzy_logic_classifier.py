#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import numpy
import pylab
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

    def read_data(self, filepath):
        """
        """
        try:
            fd = open(filepath, 'r')
            lines = fd.readlines()
            self.data = [map(float, x.strip().split(',')[1:]) for x in lines]
            self.label = [map(float, x.strip().split(',')[0]) for x in lines]
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
    
    def generate_membership_functions(self, divisions=4, do_not_use_prop=0.3):
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
        # seq = random.sample(range(0, len(self.training_data)), rules_to_generate)
        # we choose indexes for creation of initial rules.
        seq = range(rules_to_generate)
        self.number_of_classes = numpy.max(self.training_label)
        self.number_of_rules = number_of_rules
        
        # create new training data for rule generation 
        training_data = []
        for index in seq:
            training_data.append(self.training_data[index])

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
                    print "Pattern %.2f dla funkcji (%.2f, %.2f), wartosc %.2f" % (pattern[a_n], self.MF[a_n][mf_n][1], self.MF[a_n][mf_n][2], result[mf_n][0])

                den = sum(result)[0]
                if den == 0.0:
                    den = 1.0
                B_k = filter(lambda x: x[0]>0, result)
                B_k = map(lambda (x,y): [x/den,y], B_k)
                B_k = sorted(B_k, key=lambda x:x[0], reverse=True)

#                index = -1
#                r, s = random.random(), 0
#                for num in B_k:
#                    s += num[0]
#                    if s >= r:
#                        index = int(num[1])
#                        break
#
#                assert(not index == -1)
                index = int(B_k[0][1])
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
        for individual in rules:
            result = self.__train_individual(individual, self.training_data, self.training_label)
            if not result[-1][0] == self.NOT_USED:
                self.population.append(result)
        self.population = self.population[0:self.population_size]
        print len(self.population)

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

        # train each rule to obtain reliable results\
        new_rules = []
        for individual in rules:
            rule = self.__train_individual(individual, self.data_to_train, self.label_to_train)
            if not rule[-1][0] == self.NOT_USED:
                new_rules.append(rule)
        if len(new_rules):
            self.extra_rules.extend(new_rules)
            
        self.data_to_train = []
        self.label_to_train = []

    def match(self, pattern, rule):
        """
        responsible for checking IF-THEN condition in the given rule using 
        pattern.
        """

        attr_length = self.get_number_of_attributes()
        rule_value = []
        for a_n in range(attr_length):
            vMF = self.MF[a_n][rule[a_n]]
            if vMF[0] == self.USED:
                if self.__is_in_MF_range(pattern[a_n], vMF[1], vMF[2], a_n):
                    rule_value.append(self.__get_MF_value(pattern[a_n], vMF[1], vMF[2], a_n))
                else:
                    rule_value = []
                    break

        if len(rule_value):
            return [numpy.min(rule_value)*rule[attr_length][2], rule[attr_length][1]] 
        return [0, 0]
    
    def classify_pattern(self, pattern, label, rule_set):
        number_of_rules = len(rule_set)
        values = numpy.zeros((number_of_rules, 2))
        for rule_number in range(number_of_rules):
            rule = rule_set[rule_number]
            values[rule_number] = self.match(pattern, rule)

        max_alpha = values.max(axis=0) 
        numbers = filter(lambda x: x[0]>=max_alpha[0], values)
        if len(numbers) and len(filter(lambda x: not x[1]==numbers[0][1], numbers))==0 and not numbers[0][0] == 0:
            numbers = numbers[0]
            #print "Klasyfikuje obiekt %d jako %d" % (numbers[1], label)
            if int(numbers[1]) == label:
                # pattern correctly classified
                return 1
        # misclassification
        return 0
    
    def classify_by_pittsburgh(self, patterns, labels):
        classified = numpy.zeros(len(self.population))
        self.data_to_train = []
        self.label_to_train = []

        for i in range(len(self.population)):
            for j in range(len(patterns)):
                if self.classify_pattern(patterns[j], labels[j], self.population[i]):
                    classified[i] += 1
                else:
                    self.data_to_train(patterns[j])
                    self.label_to_train(labels[j])
                
            if self.the_best < classified[i]:
                self.the_best = classified[i]
                self.the_best_rule_set = self.population[i]

        print "Number of objects to classify %d " % len(labels)
        print "Recognized objects per fuzzy-set"
        counter = 0
        for number in classified:
            print "Fuzzy set %d %d" % (counter, number)
            counter += 1

        print "The best found value is %d" % self.the_best

    def classify_by_mitchigan(self, patterns, labels):
        classified = 0
        self.data_to_train = []
        self.label_to_train = []
        
        for j in range(len(patterns)):
            if self.classify_pattern(patterns[j], labels[j], self.population):
                classified += 1
            else:
                self.data_to_train(patterns[j])
                self.label_to_train(labels[j])

        if self.the_best < classified:
            self.the_best = classified

        print "Number of objects to classify %d " % len(labels)
        print "Recognized objects per fuzzy-set %d" % classified

    def run(self):
        while self.generations:
            print "Generation %d" % self.generations
            self.__create_next_generation() 
            self.generations -= 1
        self.__print_summary()
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.the_best_population)
        

#            if random.random() > 0.3:#self.mitchigan_prop:
#                indexes = random.sample(range(len(self.population)), 15)
#                for index in indexes:
#                    rule_set = self.population[index]
#                    self.__apply_crossover_for_mitchigan(rule_set)
#                    self.__apply_mutation_for_mitchigan(rule_set)
#            self.classify(self.training_data, self.training_label)

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
            return -1.0/b*value + 1.0/b*(a+b)
        
        if value >= (a-b) and value < a:
            return 1.0/b*value - 1.0/b*(a-b)
        return 0

    def __apply_crossover_for_pittsburgh(self):
        """
        """
        # select mother individual from population
        size = len(self.population)
        tab = random.sample(range(size), 3)
        mother_index = tab[0]
        father_index = tab[1]
        child_index = tab[2]
        
        # now apply crossover
        mother = self.population[mother_index]
        father = self.population[father_index]
        child = self.population[child_index]
            
        # select point of crossover
        point = random.randint(0, self.functions_per_attribute-1)
        
        if random.randint(0, 1) == 0:
            child[0:point] = mother[0:point]
            child[point:] = father[point:]
        else:
            child[0:point] = father[0:point]
            child[point:] = mother[point:]

    def __apply_crossover_for_mitchigan(self, rule_set):
        """
        """

        size = len(rule_set)
        tab = random.sample(range(size), 2)
        
        mother = rule_set[tab[0]]
        father = rule_set[tab[1]]
        child = [0]*len(mother)

        for i in range(len(mother)):
            if random.randint(0, 1):
                child[i] = mother[i]
            else:
                child[i] = father[i]
        number = len(filter(lambda x: not x==(self.functions_per_attribute - 1), child[0:-1]))
        if number == 0:
            index = random.sample(range(self.get_number_of_attributes()), 1)[0]
            child[index] = random.randint(0, self.functions_per_attribute-2)
           
        child = self.__train_individual(child, self.training_data, self.training_label)
        if not child[-1][0] == self.NOT_USED:
            self.extra_rules.append(child)  

    def __apply_mutation_for_mitchigan(self, rule_set):
        """
        """
 
        if random.random() > self.mutation_prop:
            which_rule = random.sample(range(len(rule_set)), 1)[0]
            rule = rule_set[which_rule]
            new_rule = [0]*len(rule)
            for i in range(len(rule)):
                new_rule[i] = rule[i]

            a_indexes = random.sample(range(self.get_number_of_attributes()), self.get_number_of_attributes()/2)
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
                self.extra_rules.append(individual)

    def __create_next_generation(self):
        """
        """

        self.extra_rules = []
        self.data_to_train = []
        self.label_to_train = []
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.population)
        self.__create_additional_rules()

        for _ in range(20):
            self.__apply_crossover_for_mitchigan(self.population)
            self.__apply_mutation_for_mitchigan(self.population)
        self.__evaluate_population_for_mitchigan(self.training_data, self.training_label, self.extra_rules)

        new_population = []
        new_population.extend(self.population)
        new_population.extend(self.extra_rules)

        attr_length = self.get_number_of_attributes()
        new_population = sorted(new_population, key=lambda x: x[attr_length][0], reverse=True)
        required_size = len(self.population)
        self.population = new_population[0:required_size]
        
        the_best_population = []
        the_best_population.extend(self.population)
        if len(self.the_best_population):
            the_best_population.extend(self.the_best_population)
        the_best_population = sorted(the_best_population, key=lambda x: x[attr_length][0], reverse=True)
        # how many individual we need.
        _sum = reduce(lambda x, y: x+y[-1][0], the_best_population, 0)
        _avg = _sum*1.0/len(the_best_population)
        self.the_best_population = filter(lambda x: x[-1][0]>_avg, the_best_population)
        

    def __evaluate_population_for_mitchigan(self, patterns, labels, rule_set):
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
            if final_classification[i][0] == 0 and final_classification[i][1] == 0:
                strength = -1000
            else:
                strength = 5*final_classification[i][0] - 10*final_classification[i][1]
            #rule_set[i][attr_length][0] = strength+ 2*activated[i] + 4*(1.0/number_of_attributes) + 2*rule_set[i][attr_length][2]
            rule_set[i][attr_length][0] = final_classification[i][2]+ 2*activated[i] + 4*(1.0/number_of_attributes) + 2*rule_set[i][attr_length][2]
            if self.the_best < final_classification[i][0]:
                self.the_best = final_classification[i][0]
        print "Liczba obiektow do rozpoznania %d" % len(labels)
        res = final_classification.sum(axis=0)
        if self.the_best < res[0]:
            self.the_best = res[0]
            self.the_classification = rule_set
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
            individual[attr_length][0] = self.NOT_USED
            individual[attr_length][1] = 0
            individual[attr_length][2] = 0 
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
                print "%.2f, %.2f" % (__beta_difference, __beta_total)
                strength = (beta[label] - __beta_difference)/__beta_total
                if strength > 0 :
                    individual[attr_length][0] = self.USED
                    individual[attr_length][1] = label
                    individual[attr_length][2] = strength
                    return individual
        # this is a dummy rule because has max value 
        # for different classes
        individual[attr_length][0] = self.NOT_USED
        individual[attr_length][1] = 0
        individual[attr_length][2] = 0
        return individual  

if __name__ == '__main__':
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data_new.csv')
    # fuzzy = FuzzyLogicClassifier('/home/mejcu/Pulpit/wine.data.txt')
    
    fuzzy = FuzzyLogicClassifier()
    filename = '/home/mejcu/Pulpit/wine.data.txt'
    #filename = 'wine.data.txt'
    fuzzy = FuzzyLogicClassifier(False)
    if fuzzy.read_data(filepath=filename) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=1)
    fuzzy.generate_membership_functions()
    fuzzy.initialize_genetic(population_size=20, generations=1000, mutation=0.3, crossover=0.9, mitchigan=0.5)
    fuzzy.create_population_for_mitchigan(number_of_rules=1)
    fuzzy.run()
    sys.exit(0)