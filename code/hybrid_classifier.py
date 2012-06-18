#!/usr/bin/python
# -*- coding: utf-8 -*-

from genetic_fuzzy_logic_classifier import GeneticFuzzyLogicClassifier
from genetic_rough_sets_classifier import GeneticRoughSetsClassifier
import logging
import numpy
import sys
from consts import *

if __name__ == '__main__':
    
    DEBUG = False
    result_file = 'results/hybrid_classifier.csv'
    try:
        fd = open(result_file, 'w')
    except IOError:
        print "Wrong path for results file"
        sys.exit(1)        
    
    for d in range(len(datasets)):
        for k in range(K_FOLD_NUMBER):
            for r in range(1):
                print "STARTED iteration %d for %d-fold %s dataset" % (r, k, datasets[d][0])
                fuzzy_logic_classifier = GeneticFuzzyLogicClassifier()
                rough_set_classifier = GeneticRoughSetsClassifier()
                
                filename = 'datasets/%s' % datasets[d][0]
                result = fuzzy_logic_classifier.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1]))
                result = rough_set_classifier.read_data(filepath=filename, label_is_last=(bool)(datasets[d][1]))
                if not result:
                    logging.error("Wrong dataset name")
                    sys.exit(1)
                            
                fuzzy_logic_classifier.prepare_data(k_fold_number=K_FOLD_NUMBER)
                rough_set_classifier.prepare_data(k_fold_number=K_FOLD_NUMBER)
                            
                fuzzy_logic_classifier.k_fold_cross_validation(k=k)
                fuzzy_logic_classifier.generate_membership_functions(divisions=DIVISIONS)
                fuzzy_logic_classifier.initialize_genetic(generations=200, mutation=MUTATION_PROP, crossover=CROSS_OVER_PROP)
                fuzzy_logic_classifier.create_population(population_size=20)
                fuzzy_logic_classifier.run()
                                
                rough_set_classifier.k_fold_cross_validation(k=k)
                rough_set_classifier.initialize_genetic(generations=200, mutation_prop=MUTATION_PROP, crossover_prop=CROSS_OVER_PROP)
                rough_set_classifier.create_population(population_size=POPULATION_SIZE, division=MAX_GRANULATION)
                rough_set_classifier.run()
                                
                # classify
                size = len(fuzzy_logic_classifier.testing_data)
                classified = 0
                for i in range(size):
                    ret = rough_set_classifier.classify(fuzzy_logic_classifier.testing_data[i])
                    if ret == PATTERN_REJECTED:
                        if fuzzy_logic_classifier.testing_label[i] == fuzzy_logic_classifier.classify(fuzzy_logic_classifier.testing_data[i]):
                            classified += 1
                    elif ret == fuzzy_logic_classifier.testing_label[i]:
                        classified += 1
                print "FINISHED iteration %d for %d-fold %s dataset" % (r, k, datasets[d][0])
                fd.write("%s,%d,%d,%d,%d\n" % (datasets[d][0], r, k, size, classified))
                fd.flush()
    fd.close()    