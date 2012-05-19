#!/usr/bin/python
# -*- coding: utf-8 -*-

from fuzzy_logic_classifier import FuzzyLogicClassifier
from rough_set_classifier import RoughSetClassifier
import logging
import sys

if __name__ == '__main__':
    fuzzy_logic_classifier = FuzzyLogicClassifier()
    rough_set_classifier = RoughSetClassifier()
    
    CROSS_VALIDATION_TYPE = 2
    POPULATION_SIZE = 10
    GENERATIONS = 500
    MUTATION_PROP = 0.3
    CROSS_OVER_PROP = 0.8
    FUZZY_LOGIC_DIVISIONS = 2
    ROUGH_SET_DIVISIONS = 6
    
    # read file where we store information about
    try: 
        fd = open('datasets/datatsets.txt', 'r')
        lines= fd.readlines()
        fd.close()
        datasets = [line.strip().split(' ') for line in lines]
        for data in datasets:
            data[1] = bool(int(data[1]))
        
        for dataset in datasets:
            result = fuzzy_logic_classifier.read_data(filepath=dataset[0], label_is_last=dataset[1])
            result = rough_set_classifier.read_data(filepath=dataset[0], label_is_last=dataset[1])
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
                
                
            
            
            
        
    except IOError:
        logging.error("Wrong path for file with datasets")
        sys.exit(1)
    
    