#! /bin/python
# *-* encoding=utf-8 *-*

import numpy
import pylab
import random
import struct
import math

def float_to_bin(f):
    """ convert float to binary string """
    ba = struct.pack('>d', f)
    s = ''.join('{0:08b}'.format(ord(b)) for b in ba)
    # strip off leading zeros
    for i in range(len(s)):
        if s[i] != '0':
            break
    else:  # all zeros
        s = '0'
        i = 0
    return s[i:]

def int_to_bytes(n, minlen=0):  # helper function
    """ int/long to byte string """
    nbits =  int(math.ceil(math.log(n, 2)))+ (1 if n < 0 else 0)  # plus one for any sign bit
    nbytes = (nbits+7)/8  # number of whole bytes
    bytes = []
    for i in range(nbytes):
        bytes.append(chr(n & 0xff))
        n >>= 8
    # zero pad if needed
    if minlen > 0 and len(bytes) < minlen:
        bytes.extend((minlen-len(bytes)) * '0')
    bytes.reverse()  # put high bytes at beginning
    return ''.join(bytes)

def bin_to_float(b):
    """ convert binary string to float """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64
    return struct.unpack('>d', bf)[0]

if __name__ == '__main__':
    
#    basic_x = [-2.9 , -1.5, -1.0 , -0.5 ,0, 0.5, 1.0, 1.5, 2.9]
#    MFs = 10
#    for i in range(0, 7):
#        mi = random.randint(3, 45)
#        sigma = random.random()*random.randint(2, 20)
#        x = [_x*sigma+mi for _x in basic_x]
#        y = [numpy.exp((-(_x-mi)*(_x-mi)/(2*sigma*sigma))) for _x in x ]
#        pylab.plot(x, y)
#        
#    # additionally we want to have 2 MF on the end
#    mi = 0
#    sigma = 10
#    x = [_x*sigma+mi for _x in basic_x]
#    y = [numpy.exp((-(_x-mi)*(_x-mi)/(2*sigma*sigma))) for _x in x ]
#    pylab.plot(x, y)
#    
#    mi = 50
#    sigma = 10
#    x = [_x*sigma+mi for _x in basic_x]
#    y = [numpy.exp((-(_x-mi)*(_x-mi)/(2*sigma*sigma))) for _x in x ]
#    pylab.plot(x, y)
#        
#    pylab.xlabel('x')
#    pylab.ylabel('y')    
#    pylab.grid(True)
#    pylab.show()
    
    floats = [1.0, -14.0, 12.546]

    for f in floats:
        binary = float_to_bin(f)

        print 'float_to_bin(%f): %r' % (f, binary)
        float = bin_to_float(binary)
        print 'bin_to_float(%r): %f' % (binary, float)

    
    #t = numpy.arange(-15.0, 15.0, 0.05)
    #s = numpy.exp(-(t-mi)*(t-mi)/(2*sigma*sigma))
    #pylab.plot(t, s)
    
    
    
