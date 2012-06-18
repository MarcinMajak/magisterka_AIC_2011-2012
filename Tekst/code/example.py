#!/usr/bin/python
# -*- coding: utf-8 -*- 

from genetic_rough_sets_classifier import GeneticRoughSetsClassifier

if __name__ == '__main__':
    K_FOLD_NUMBER = 4
    K = 1
    GENERATIONS = 500
    MUTATION = 0.3
    CROSS_OVER = 0.9
    G = 7
    POPULATION = 10
    grs = GeneticRoughSetsClassifier(debug=DEBUG)
    filename = 'datasets/iris.data.txt'
    grs.read_data(filepath=filename, label_is_last=False)
    grs.prepare_data(k_fold_number=K_FOLD_NUMBER)
    grs.k_fold_cross_validation(k=K)
    grs.initialize_genetic(generations=GENERATIONS, mutation_prop=MUTATION, crossover_prop=CROSS_OVER)
    grs.create_population(population_size=POPULATION, division=G)
    # run genetic algorithm
    grs.run()
    # genetic algorithm found decision rule set
    # classify patterns
    for pattern in patterns:
        grs.classify(pattern)
