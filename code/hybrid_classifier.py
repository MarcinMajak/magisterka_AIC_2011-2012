#!/usr/bin/python
# -*- coding: utf-8 -*-

from genetic_fuzzy_logic_classifier import GeneticFuzzyLogicClassifier
from genetic_rough_sets_classifier import GeneticRoughSetsClassifier
import logging
import numpy
import sys

if __name__ == '__main__':
    fuzzy_logic_classifier = GeneticFuzzyLogicClassifier()
    rough_set_classifier = GeneticRoughSetsClassifier()
    
    CROSS_VALIDATION_TYPE = 2
    POPULATION_SIZE = 10
    GENERATIONS = 500
    MUTATION_PROP = 0.3
    CROSS_OVER_PROP = 0.8
    FUZZY_LOGIC_DIVISIONS = 2
    ROUGH_SET_DIVISIONS = 6
    RUNS = 5
    
    # read file where we store information about
    try: 
        fd = open('datasets/datasets.txt', 'r')
        lines= fd.readlines()
        fd.close()
        datasets = [line.strip().split(' ') for line in lines]
        for data in datasets:
            data[1] = bool(int(data[1]))
        
        for dataset in datasets:
            for run in range(RUNS):
                result = fuzzy_logic_classifier.read_data(filepath="datasets/%s" % dataset[0], label_is_last=dataset[1])
                result = rough_set_classifier.read_data(filepath="datasets/%s" % dataset[0], label_is_last=dataset[1])
                if not result:
                    logging.error("Wrong dataset name")
                    continue
                
                fuzzy_logic_classifier.prepare_data(k_fold_number=CROSS_VALIDATION_TYPE)
                rough_set_classifier.prepare_data(k_fold_number=CROSS_VALIDATION_TYPE)
                
                for k_fold_number in range(CROSS_VALIDATION_TYPE):
                    fuzzy_logic_classifier.k_fold_cross_validation(k=k_fold_number)
                    fuzzy_logic_classifier.generate_membership_functions(divisions=FUZZY_LOGIC_DIVISIONS)
                    fuzzy_logic_classifier.initialize_genetic(generations=GENERATIONS, mutation=MUTATION_PROP, crossover=CROSS_OVER_PROP)
                    fuzzy_logic_classifier.create_population(population_size=POPULATION_SIZE)
                    fuzzy_logic_classifier.run()
                    
                    rough_set_classifier.k_fold_cross_validation(k=k_fold_number)
                    rough_set_classifier.k_fold_cross_validation(k=k_fold_number)
                    rough_set_classifier.initialize_genetic(generations=GENERATIONS, mutation_prop=MUTATION_PROP, crossover_prop=CROSS_OVER_PROP)
                    rough_set_classifier.create_population(population_size=POPULATION_SIZE, division=ROUGH_SET_DIVISIONS)
                    rough_set_classifier.run()
                    
                    # classify
                    size = len(fuzzy_logic_classifier.testing_data)
                    classification = numpy.zeros((size, 2))
                    for i in range(size):
                        if fuzzy_logic_classifier.testing_label[i] == fuzzy_logic_classifier.classify(fuzzy_logic_classifier.testing_data[i]):
                            classification[i][0] = 1
                        val = rough_set_classifier.classify(fuzzy_logic_classifier.testing_data[i])
                        if fuzzy_logic_classifier.testing_label[i] == val:
                            classification[i][1] = 1
                        else:
                            print "Nie zaklasyfikowalo poniewaz zwrocilo %d" % val
                    res = classification.sum(axis=0)
                    logging.debug("Fuzzy %d, rough %d out of %d" % (res[0], res[1], size))
                    # save results
    except IOError:
        logging.error("Wrong path for file with datasets")
        sys.exit(1)
    
    