# -*- coding: utf-8 -*-
from mnist import MNIST


mndata = MNIST("/home/rw/codeplace/regression/mnist")
data = mndata.load_training()
labels = data[1]
features = data[0]

def get_3_5():
    fs = []
    ls = []
    for index, l in enumerate(labels):
        if l == 3:
            ls.append(1)
        if l == 5:
            ls.append(-1)
        if l == 3 or l == 5:
            fs.append(features[index])
    return fs, ls 
