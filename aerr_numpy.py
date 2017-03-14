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


class AERR(object):

    def __init__(self, k, B, lr):
        self.k = k
        self.B = B
        self.lr = lr
   
    def predict(self, X):
        return self.avg_w.dot(X)

    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1

            
    def train(self, fs, ls):
        B = self.B
        d = len(fs[0])
        w = 0.0001 * numpy.random.random(d)
        ws = [w]
        k = self.k
        m = len(fs)
        self.lr = (k/2.0*d*m) ** 0.5
        self.indexs = [i for i in range(d)]
        for index, x in enumerate(fs):
            y = ls[index]
            w = ws[-1]
            x_t = numpy.zeros(d)
            for i in range(k):
                x_index = numpy.random.choice(self.indexs)
                x_t[x_index] += d * x[x_index]
            x_t = x_t / k
            w_norm = numpy.linalg.norm(w, 2) ** 2
            percents = w * w / w_norm
            w_index = numpy.random.choice(self.indexs, p=percents)
            phi = w_norm * x[w_index]/ w[w_index] - y
            g = phi * x_t
            v = w - self.lr * g
            new_w = v * B / max(B, numpy.linalg.norm(v, 2))
            ws.append(new_w)
        self.avg_w = sum(ws) / m


if __name__ == '__main__':
    fs, ls = mnist_data.get_3_5()
    r = AERR(56, 0.45, 0)
    k_fold.k_fold(fs, ls, 10, r)
