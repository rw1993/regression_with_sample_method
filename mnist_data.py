# -*- coding: utf-8 -*-
from mnist import MNIST


mndata = MNIST("/home/rw/codeplace/regression_with_sample_method/mnist")
data = mndata.load_training()
labels = data[1]
features = data[0]

def one_or_zero(x):
    return 1.0 if x > 0 else -1.0

def get_3_5():
    fs = []
    ls = []
    for index, l in enumerate(labels):
        if l == 3:
            ls.append(-1.0)
        if l == 5:
            ls.append(1.0)
        if l == 3 or l == 5:
            fs.append(map(one_or_zero, features[index]))
            #fs.append(features[index])
    return fs, ls 

if __name__ == '__main__':
    fs , ls = get_3_5()
    print fs[0]
