#!/usr/bin/python
# -*- coding: utf-8 -*- 

from genetic_rough_sets_classifier import GeneticRoughSetsClassifier

if __name__ == '__main__':
    grs = GeneticRoughSetsClassifier(debug=DEBUG)
    filename = 'datasets/thyroid.data.txt'
    grs.read_data(filepath=filename, label_is_last=False)
    grs.prepare_data(k_fold_number=K_FOLD_NUMBER)
    grs.k_fold_cross_validation(k=K)
    grs.initialize_genetic(generations=500, mutation_prop=0.3, crossover_prop=0.9)
    grs.create_population(population_size=10, division=7)
    grs.run()
    for pattern in patterns:
        grs.classify(pattern)
