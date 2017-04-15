import numpy
import arma


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


class AELR(object):

    def __init__(self, k, B=32, lr=2**-7, mr=0.0):
        self.k = k
        self.B = B
        self.lr = lr
        self.mr = mr

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
        ws = []
        k = self.k
        m = len(fs)
        self.indexs = [i for i in range(d)]
        z_p = numpy.ones(d)
        z_n = numpy.ones(d)
        for index, x in enumerate(fs):
            y = ls[index]
            if y == "*":
                continue
            indexs = [index for index, x_ in enumerate(x) if x != "*"]
            x = [x_ if x_ != '*' else 0.0 for x_ in x]
            w = (z_p - z_n) * B / (numpy.linalg.norm(z_p, 1) + numpy.linalg.norm(z_n, 1))
            ws.append(w)
            mr = float(len(indexs)) / d
            k1 = int(k*mr)
            x_t = numpy.zeros(d)
            for r in range(k1):
                x_index = numpy.random.choice(indexs)
                x_t[x_index] += d * x[x_index]
            x_t = x_t / k1
            w_norm = numpy.linalg.norm(w, 1)
            if w_norm == 0:
                phi = -y
            else:
                ps = abs(w) / w_norm
                w_index = numpy.random.choice(self.indexs, p=ps)
                phi = w_norm * numpy.sign(w[w_index]) * x[w_index] / (1-self.mr) - y
            g = phi * x_t
            for i in range(d):
                g[i] = max(min(g[i], 1.0/self.lr), -1.0/self.lr)
                z_p[i] *= numpy.exp(-self.lr*g[i]*mr)
                z_n[i] *= numpy.exp(self.lr*g[i]*mr)
        self.avg_w = sum(ws) / m


def prepare_ar(lenth, missing_rate):
    #a = arma.ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
    a = arma.ARMA([0.11, -0.5], [], 0.5, noise_type="uni")
    p = a.p
    time_series = [a.generater.next() for i in range(lenth)]
    train_lenth = int(len(time_series)*0.7)
    train_time_series = time_series[:train_lenth]
    test_time_series = time_series[train_lenth:]
    for index, t in enumerate(train_time_series):
        if numpy.random.random() < missing_rate:
            train_time_series[index] = '*'
    train_fs = []
    train_ls = []
    test_fs = []
    test_ls = []
    for index, t in enumerate(train_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            train_fs.append(f)
            train_ls.append(l)
        except:
            break
    for index, t in enumerate(test_time_series):
        try:
            l = time_series[index+p]
            f = time_series[index: index+p]
            test_fs.append(f)
            test_ls.append(l)
        except:
            break
    return train_fs, train_ls, test_fs, test_ls, train_time_series


if __name__ == '__main__':
    best = (99999999, 0, 0)
    Bs = [2**i for i in range(-6, 6)]
    lrs = [2**i for i in range(-9, 9)]
    # B=32, lr=2**-7 missing = 0.0
    # B=16, lr=2**-7 missing = 0.1 0.0919
    # B=4, lr=2**-4 missing = 0.2 0.0925
    '''
    for B in Bs:
        for lr in lrs:
            try:
                mses = []
                for i in range(20):
                    r = AELR(2, B, lr, mr=0.3)
                    train_fs, train_ls, test_fs, test_ls = prepare_ar(10000,
                                                                      0.3)
                    r.train(train_fs, train_ls)
                    errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
                    mse = sum(e*e for e in errors) / len(errors)
                    mses.append(mse)
                avg_mse = sum(mses) / len(mses)
                best = (avg_mse, lr, B) if best[0] > avg_mse else best
            except Exception, e:
                print e
                print B, lr, "failed"
            print best
    '''
    for mr in [0.0, 0.1, 0.2, 0.3]:
        mses = []
        for i in range(20):
            #r = AELR(5, 32, 2**-7, mr)
            r = AELR(2, 32, 2**-7, mr)
            train_fs, train_ls, test_fs, test_ls, _ = prepare_ar(10000,
                                                              mr)
            r.train(train_fs, train_ls)
            errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
            mse = sum(e*e for e in errors) / len(errors)
            mses.append(mse)
        avg_mse = sum(mses) / len(mses)
        print avg_mse

    '''
    mses = []
    for i in range(20):
        print i
        #r = AERR(2, 1, 0.0625, mr)
        mr = 0.4 * numpy.random.random()
        train_fs, train_ls, test_fs, test_ls, train_time_series = prepare_ar(10000,
                                                                             mr)
       
        mr = sum(map(lambda x: 1.0 if x == '*' else 0.0,
                     train_time_series[:1000])) / 1000
        r = AELR(5, 1, 0.0625, mr)
        #r = AELR(2, 1, 0.0625, mr)
        r.train(train_fs, train_ls)
        errors = [r.predict(x)-y for x, y in zip(test_fs, test_ls)]
        mse = sum(e*e for e in errors) / len(errors)
        mses.append(mse)
    avg_mse = sum(mses) / len(mses)
    print avg_mse
    '''
