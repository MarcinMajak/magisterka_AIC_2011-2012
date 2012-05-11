#!/usr/bin/python
# -*- coding: utf-8 -*- 

import random
import math
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
                
                if self.DEBUG:
                    figure = pylab.figure(i)
                
                div = float((self.max[a_n] - self.min[a_n])*1.0/(i-1))
                sigma = div/3.0
                
                mi = self.min[a_n]
                self.MF[a_n].append([self.USED, mi, sigma])
                
                if self.DEBUG:
                    (x, y) = self.__get_characteristic_points(mi, sigma, a_n)
                    pylab.plot(x, y)
                
                for j in range(1, i-1):
                    mi = float(self.min[a_n] + j*1.0*div)
                    self.MF[a_n].append([self.USED, mi, sigma])
                    
                    if self.DEBUG:
                        (x, y) = self.__get_characteristic_points(mi, sigma, a_n)
                        pylab.plot(x, y)                
    
                mi = self.max[a_n]
                self.MF[a_n].append([self.USED, mi, sigma])
                
                if self.DEBUG:
                    (x, y) = self.__get_characteristic_points(mi, sigma, a_n)
                    pylab.plot(x, y)
            # this denotes don't care function
            self.MF[a_n].append([self.NOT_USED, 0, 0])
            
            if self.DEBUG:
                pylab.grid(True)
                pylab.show()

    def intialize_genetic(self, population_size, generations, mutation, crossover):
        """
        Sets population size, number of generations, mutation and crossover 
        probabilities
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_prop = mutation
        self.crossover_prop = crossover

    def create_population(self, number_of_rules, number_of_classes=2):
        """
        """
        if (number_of_rules+10) > len(self.training_label):
            rules_to_generate = len(self.training_label)
        else:
            rules_to_generate = number_of_rules + 10
            
        seq = random.sample(range(0, len(self.training_data)), rules_to_generate)
        self.number_of_classes = number_of_classes
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
                    result[mf_n][0] = self.__get_function_value(pattern[a_n], self.MF[a_n][mf_n][1], self.MF[a_n][mf_n][2], a_n, False)
                    result[mf_n][1] = mf_n
            
                den = sum(result)[0]
                if den == 0.0:
                    den = 1.0
                B_k = filter(lambda x: x[0]>0, result)
                B_k = map(lambda (x,y): [x/den,y], B_k)
                B_k = sorted(B_k, key=lambda x:x[0])

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
                numbers = random.sample(range(attr_length), int(random.uniform(0.1, 0.3)*attr_length))
                for number in numbers:
                    # the last index in the MF functions indicates don't care
                    rules[rule_number][number] = attr_length
                    
        # train each rule to obtain reliable results 
        for individual in rules:
            self.__train_individual(individual)
            
        # now we have trained rules
        self.population = []
        for i in range(self.population_size):
            self.population.append([])
            seq = random.sample(range(0, rules_to_generate), self.number_of_rules)
            for index in seq:
                self.population[i].append(rules[index])

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
        pass
    
    def match(self, pattern, rule):
        """
        is responsible for checking IF-THEN condition in the given rule using 
        pattern.
        """
        
        attr_length = self.get_number_of_attributes()        
        rule_value = []
        for a_n in range(attr_length):
            vMF = self.MF[a_n][rule[a_n]]
            if vMF[0] == self.USED:
                (x, y) = self.__get_characteristic_points(vMF[1], vMF[2], a_n)
                if (pattern[a_n]>=x[0] and pattern[a_n]<=x[8]):
                    rule_value.append(self.__get_function_value(pattern[a_n], vMF[1], vMF[2], a_n))    
                else:
                    rule_value = []
                    break
                                
        if len(rule_value):
            return [numpy.min(rule_value)*rule[attr_length][2], rule[attr_length][1]] 
        return [0, 0]
    
    def classify_pattern(self, pattern, label, rule_set):
        values = numpy.zeros((self.number_of_rules, 2))
        for rule_number in range(self.number_of_rules):
            rule = rule_set[rule_number]
            values[rule_number] = self.match(pattern, rule)
            
        max_alpha = values.max(axis=0) 
        numbers = filter(lambda x: x[0]>=max_alpha[0], values)
        if len(numbers) and len(filter(lambda x: not x[1]==numbers[0][1], numbers))==0 and not numbers[0][0] == 0:
            numbers = numbers[0]
            print "Klasyfikuje obiekt %d jako %d" % (numbers[1], label)
            if int(numbers[1]) == label:
                # pattern correctly classified
                return 1
        # misclassification
        return 0
    
    def classify(self, patterns, labels):
        classified = numpy.zeros(len(self.population))
        for i in range(len(self.population)):
            for j in range(len(patterns)):
                classified[i] += self.classify_pattern(patterns[j], labels[j], self.population[i])
                
        print "Number of objects to classify %d " % len(labels)
        print "Recognized objects per fuzzy-set"
        counter = 0
        for number in classified:
            print "Fuzzy set %d %d" % (counter, number)
            counter += 1     
            
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
    
    def __get_characteristic_points(self, mean, sigma, which, check_boundaries=True):
        points = [-3.0 , -2.0, -1.0 , -0.5 ,0, 0.5, 1.0, 2.0, 3.0]
        x = [_x*sigma+mean for _x in points]
        y = [numpy.exp((-(_x-mean)*(_x-mean)/(2*sigma*sigma))) for _x in x ]
    
        print "Zakres: [%.2f, %.2f], wyszlo: [%.2f, %.2f] dla [%.2f, %.2f]" % \
                    (self.min[which], self.max[which],
                     x[0], x[8],
                     mean, sigma),
    
        """
        Here we check where function is placed.
        """
        if check_boundaries and x[0] <= self.min[which]:
            print " FUNKCJA JEST POZA LEWYM ZAKRESEM\n"
            x[0] = -9999.0
            y[0] = 1.0
            
        if check_boundaries and x[8] >= self.max[which]:
            print " FUNKCJA JEST POZA PRAWYM ZAKRESEM\n"
            x[8] = 9999.0
            y[8] = 1.0
      
        return (x, y)
    
    def __get_function_value(self, val, mean=0, sigma=1, which=0, check_boundaries=True):
        """
        """
        if check_boundaries and val <= self.min[which]:
            #print "JEST POZA LEWYM ZAKRESEM"
            return 1.0
            
        if check_boundaries and val >= self.max[which]:
            #print "JEST POZA PRAWYN ZAKRESEM"
            return 1.0
        
        if sigma == 0:
            return 1
        return numpy.exp(-(val-mean)**2/(2*sigma**2))
        
    def __apply_crossover(self):
        """
        """
        # select mother individual from population
        size = len(self.population)
        tab = random.sample(range(size), size)
        tab = random.sample(tab, 3)
        mother_index = tab[0]
        father_index = tab[1]
        child_index = tab[2]
        
        # now apply crossover
        mother = self.population[mother_index]
        father = self.population[father_index]
        child = self.population[child_index]
        for i in range(self.get_number_of_attributes()):
            m_mf = mother[i]
            f_mf = father[i]
            if m_mf[0] == self.USED and f_mf[0] == self.USED:
                bin_m_m = self.__convert_to_bin(m_mf[1], i)
                bin_m_sigma = self.__convert_to_bin(m_mf[2], i, True)
                
                bin_f_m = self.__convert_to_bin(f_mf[1], i)
                bin_f_sigma = self.__convert_to_bin(f_mf[2], i, True)
                
                # select point of crossover
                point = random.randint(1, len(bin_f_m)-1)
                if random.randint(0, 1) == 0:
                    child_m = bin_m_m[0:point] + bin_f_m[point:]
                    child_sigma = bin_m_sigma[0:point] + bin_f_sigma[point:]
                else:
                    child_m = bin_f_m[0:point] + bin_m_m[point:]
                    child_sigma = bin_f_sigma[0:point] + bin_m_sigma[point:]
            
                m_val = self.__convert_to_real(child_m, i)
                sigma_val = self.__convert_to_real(child_sigma, i, True)
            
                child[i][0] = self.USED
                child[i][1] = m_val
                if not sigma_val == 0: 
                    child[i][2] = sigma_val
        
    def __convert_to_bin(self, real_value, which, is_sigma=False, bits_number=16):
        """
        """
        min_range = self.min[which]
        max_range = self.max[which]
        if is_sigma:
            max_range = float((max_range - min_range)*1.0/(self.number_of_variables))
            min_range = 0.1
            
        bin_value = (2**bits_number - 1)*(real_value-min_range)/(max_range-min_range)
        bin_value = bin(int(bin_value))
        bin_value = bin_value[2:]
        if len(bin_value) < bits_number:
            number = bits_number - len(bin_value)
            bin_value = ''.join("%d" % x for x in [0]*number) + bin_value
        return bin_value
            
    def __convert_to_real(self, bin_value, which, is_sigma=False, bits_number=16):
        """
        """
        min_range = self.min[which]
        max_range = self.max[which]
        
        if is_sigma:
            max_range = float((max_range - min_range)*1.0/(self.number_of_variables))
            min_range = 0.1
        bin_value = int(bin_value, 2)
        return float(min_range + (float(bin_value*1.0)/(2**bits_number - 1)*1.0)*(max_range-min_range))
            
    def __apply_mutation(self):
        """
        """
        if random.random() > self.mutation_prop:
            index = random.randint(1, len(self.population)) - 1
            individual = self.population[index]
            for i in range(self.get_number_of_attributes()):
                individual[i][0] = self.USED if random.randint(0, 1) else self.NOT_USED
                individual[i][1] = random.triangular(self.min[i], self.max[i])
                div =  float((self.max[i] - self.min[i])/(1.0*self.number_of_variables))
                individual[i][2] = random.uniform(0.2, div)/2.0
            
    def __create_next_generation(self):
        """
        """
        val = reduce(lambda a,d: a+d, self.fitness_index, 0)
        avg = val*1.0/len(self.fitness_index)
        # remove those individuals that are very poor
        required_size = len(self.population)
        new_population = []
        new_fitness_index = []
        for i in range(len(self.population)):
            if self.fitness_index[i] >= avg:
                new_population.append(self.population[i])
                new_fitness_index.append(self.fitness_index[i])
                
        if len(new_population) < required_size:
            indexes = range(0, len(new_population))
            number_of_new_individuals = required_size - len(new_population)
            for i in range(number_of_new_individuals):
                # select mother individual from population
                size = len(indexes)
                index = random.randint(1, size) - 1
                curr_val = indexes[index]
                indexes[index] = indexes[size - 1]
                indexes[size - 1] = curr_val
                mother_index = curr_val
                
                # select father individual from population
                size = size - 1
                index = random.randint(1, size) - 1
                curr_val = indexes[index]
                indexes[index] = indexes[size - 1]
                indexes[size - 1] = curr_val
                father_index = curr_val
                
                # now apply crossover
                mother = new_population[mother_index]
                father = new_population[father_index]
                child = []
                mod = (self.get_number_of_attributes() + 1)
                for i in range(0, len(mother)):
                    m_mf = mother[i]
                    f_mf = father[i]

                    bin_m_m = self.__convert_to_bin(m_mf[1], i%mod)
                    bin_m_sigma = self.__convert_to_bin(m_mf[2], i%mod, True)
                    
                    bin_f_m = self.__convert_to_bin(f_mf[1], i%mod)
                    bin_f_sigma = self.__convert_to_bin(f_mf[2], i%mod, True)
                    
                    # select point of crossover
                    point = random.randint(1, len(bin_f_m)-1)
                    if random.randint(0, 1) == 0:
                        child_m = bin_m_m[0:point] + bin_f_m[point:]
                        child_sigma = bin_m_sigma[0:point] + bin_f_sigma[point:]
                    else:
                        child_m = bin_f_m[0:point] + bin_m_m[point:]
                        child_sigma = bin_f_sigma[0:point] + bin_m_sigma[point:]
                
                    m_val = self.__convert_to_real(child_m, i%mod)
                    sigma_val = self.__convert_to_real(child_sigma, i%mod, True)
                
                    new_MF = [self.USED, m_val, sigma_val if not sigma_val == 0 else 0.4 ]
                    child.append(new_MF)
                fitness_index = self.__evaluate_individual(child)
                new_population.append(child)
                new_fitness_index.append(fitness_index)
        self.population = None
        self.fitness_index = None
        self.population = new_population
        self.fitness_index = new_fitness_index
    
    def __train_individual(self, individual):
        """
        Function responsible for training individual and assigning
        strength to a given rule.
        """
        attr_length = self.get_number_of_attributes()
        beta = {}
        for i in range(len(self.training_data)):
            rule_value = []
            for variable in range(0, attr_length):
                vMF = individual[variable]
                if vMF[0] == self.USED:
                    # check if function is in the range of a given membership function
                    (x, y) = self.__get_characteristic_points(vMF[1], vMF[2], variable)
                    if (self.training_data[i][variable]>=x[0] and self.training_data[i][variable]<=x[8]):
                        rule_value.append(self.__get_function_value(self.training_data[i][variable], vMF[1], vMF[2], variable))    
                    else:
                        rule_value = []
                        break
            if len(rule_value):
                min_value = numpy.min(rule_value)
                # here we store value of min product and class label
                if not self.training_label[i] in beta.keys():
                    beta[self.training_label[i]] = 0.0
                beta[self.training_label[i]] += min_value 
                    
        if len(beta.keys()) == 0:
            # this is a dummy rule.
            individual[attr_length][0] = self.NOT_USED
            individual[attr_length][1] = 0
            individual[attr_length][2] = 0 
        else:
            # now we have to decide which class we should assign to this rule
            # and additionally the corresponding strength.
            max_value = numpy.max(beta.values())
            labels = filter(lambda key: beta[key]>=max_value, beta)
            if len(labels) and len(filter(lambda x: not x==labels[0], labels))==0:
                label = labels[0]
                # calculate strength factor.
                __beta_difference = 0.0
                __beta_total = 0.0
                for key in beta:
                    if not key == label:
                        __beta_difference += beta[key]
                    __beta_total += beta[key]
                __beta_difference = float(__beta_difference/(self.number_of_classes-1)*1.0)
                strength = (beta[label] - __beta_difference)/__beta_total
                individual[attr_length][0] = self.USED
                individual[attr_length][1] = label
                individual[attr_length][2] = strength
            else:
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
    fuzzy = FuzzyLogicClassifier(False)
    if fuzzy.read_data(filename=filename) == False:
        print "Error with opening the file. Probably you have given wrong path"
        sys.exit(1)
    fuzzy.prepare_data(k_fold_number=2)
    fuzzy.k_fold_cross_validation(k=1)
    fuzzy.generate_membership_functions()
    fuzzy.intialize_genetic(population_size=20, generations=100, mutation=0.1, crossover=0.9)
    fuzzy.create_population(number_of_rules=30, number_of_classes=2)
    fuzzy.classify(patterns=fuzzy.training_data, labels=fuzzy.training_label)
    # fuzzy.run()