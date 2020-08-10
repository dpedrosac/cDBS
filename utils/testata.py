import numpy

a = [2,3,4,5]
filename = 'xx'


def mean_estimation(input):
    b = 0
    for k in input:
        b += k

    b = b/len(input)
    return b