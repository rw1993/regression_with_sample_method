import numpy
import mnist_data
import k_fold
import projection
import math
import itertoolsmodule

def one_or_one(func):

    def _f(*args, **kw):
        r = func(*args, **kw)
        if r > 1.0:
            return 1.0
        if r < -1.0:
            return -1.0
        else:
            return r
    return _f


class AER(object):

    def __init__(self, k, B, lr):
        self.k = k
        self.B = B
        self.lr = lr
   
    #@one_or_one 
    def predict(self, X):
        return self.arg_w.dot(X)

    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1

            
    def train(self, fs, ls):
        B = self.B
        d = len(fs[0])
        w = numpy.zeros(d)
        avg_w = numpy.zeros(d)
        k = self.k
        m = len(fs)
        self.lr = (B+1.0)*d/B*(math.log(m)/(m*k))**0.5
        self.indexs = [i for i in range(d)]
        for index, x in enumerate(fs):
            y = ls[index]
            t = index + 1.0
            v = numpy.zeros(d)
            C = numpy.random.choice(self.indexs, k/2)
            for j in C:
                v[j] +=  2.0 / k * d * x[j]
            e_y = .0
            w_norm = numpy.linalg.norm(w, 1)
            if w_norm == 0:
                w = 2.0/self.lr/t*y*v
            else:
                percents = abs(w) / w_norm
                indexs = [i for i in range(d)]
                for i in range(k/2):
                    w_index = numpy.random.choice(indexs, p=percents)
                    e_y += 2.0 / k * numpy.sign(w[w_index]) * w_norm * x[w_index]
                w = (1.0 - 1.0/t)*w - 2.0/self.lr/t*(e_y-y)*v
            if numpy.linalg.norm(w, ord=1) > self.B:
                w = projection.euclidean_proj_l1ball(w, self.B)

            avg_w += w/m
        self.arg_w = avg_w

if __name__ == '__main__':

    fs, ls = mnist_data.get_3_5()
    r = AER(4, 0.70, 10)
    k_fold.k_fold(fs, ls, 10, r)
