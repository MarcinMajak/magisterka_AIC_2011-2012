#!/usr/bin/python
#-*- coding:utf-8 -*-

import numpy

if __name__=='__main__':
    FILE_TO_READ = 'hybrid_classifier.csv'
    FILE_TO_WRITE = 'hybrid_done.csv'
    fd = open(FILE_TO_READ, 'r')
    lines = fd.readlines()
    fd.close()
    tab = [x.strip().split(',') for x in lines]

    STEP = 4

    fd = open(FILE_TO_WRITE, 'w')

    for i in range(len(tab)/STEP):
        curr_tab = []
        fd.write(tab[STEP*i+0][0])
        fd.write(',')

        val = []
        # calculate average number of objects
        for o in range(STEP):
            val.append(int(tab[STEP*i+o][3]))
        
        fd.write("%.3f" %  numpy.average(val))
        fd.write(',')

        # calculate min, max, average and deviation
        val = []
        # calculate average number of objects
        for o in range(STEP):
            val.append(int(tab[STEP*i+o][4]))


        fd.write("%.3f" % numpy.min(val))
        fd.write(',')
        fd.write("%.3f" % numpy.max(val))
        fd.write(',')
        fd.write("%.3f" % numpy.average(val))
        fd.write(',')
        fd.write("%.3f" % numpy.std(val))
        fd.write('\n')
    fd.close()
