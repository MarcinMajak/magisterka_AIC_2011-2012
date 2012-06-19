#!/usr/bin/python
# -*- coding: utf-8 -*- 


datasets = [
            ['iris.data.txt', 1],
            ['bupa.data.txt', 1],
            ['pima.data.txt', 1],
            ['haberman.data.txt', 1],
            ['wdbc.data.txt', 0],
            ['thyroid.data.txt', 0],
            ['wine.data.txt', 0]         
    ]

"""
Determinies the type of cross-validation type.
In this project 4-cross-validation is used which 
means that in each run dataset is divided into 4 
even sets and one set is treated as testing and
three remaining ones are taken for testing.
"""
K_FOLD_NUMBER = 4

"""
This variable indicate how many times each test is repeated
"""
REPEAT_TEST = 3

"""
This variable tells what is the maximal step for granulation in 
genetic rough sets classifier
"""
MAX_GRANULATION = 8

"""
This variable indicates how many fuzzy membership functions are generated
for feature division
"""
DIVISIONS = 4

"""
This number indicates the size of population used in genetic algorithm 
"""
POPULATION_SIZE = 10

"""
Detemines the probability of mutation operator in genetic algorithm
"""
MUTATION_PROP = 0.3

"""
Determines the probaility of cross-over operator in genetic algorithm
"""
CROSS_OVER_PROP = 0.8

"""
Determines the number of iterations for genetic algorithm
"""
GENERATIONS = 500

"""
"""
PATTERN_REJECTED = -1