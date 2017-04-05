# -*- coding:utf8 -*-
import numpy
import sys
import random

class ARMA(object, ):
    
    
    @property
    def max_noise(self):
        return max(map(abs, self.noises))


    def __init__(self, alphas, betas, sigma, noise_type="normal" ):
        self.noise_map = {"normal": self.normal_noise,
                          "uni": self.uni_noise}
        self.noise =  self.noise_map[noise_type]
        self.alphas = alphas
        self.betas = betas
        self.sigma = sigma
        self.p = len(self.alphas)
        self.init_first_few_xs()
        self.generater = self.generate_data()

    def init_first_few_xs(self):
        self.xs = [self.noise() for alpha in self.alphas]
        self.noises = [self.noise() for alpha in self.alphas]

    @property
    def current_xs(self):
        return self.xs[-len(self.alphas):]


    @property
    def current_noises(self):
        return self.noises[-len(self.betas):]
       

    def normal_noise(self):
        if self.sigma == 0:
            return 0
        return numpy.random.normal(0, self.sigma, 1)[0]
    
    def uni_noise(self):
        if self.sigma == 0:
            return 0
        return numpy.random.uniform(-self.sigma, self.sigma)

    def generate_data(self):
        while True:
            # AR part
            sum_ar = 0.0
            for alpha, x in zip(self.alphas, self.current_xs):
                sum_ar += alpha * x

            # MA part
            sum_ma = 0.0
            for beta, noise in zip(self.betas, self.current_noises):
                sum_ma += beta * noise

            n = self.noise()
            x = sum_ar + sum_ma + n

            self.xs.append(x)
            self.noises.append(n)
            '''
            if abs(x) > 1:
                if x > 0:
                    x = 1
                else:
                    x = -1
            '''
            yield x


def write(time_series, mr, time, limit=10):
    with open("./data/data_{mr}_{time}.csv".format(mr=mr, time=time),
              "w") as f:
        sys.stdout = f
        for index, time in enumerate(time_series):
            if random.random() > mr or index < limit:
                print time
            else:
                print "nan"

if __name__ == "__main__":
    mrs = [0.0, 0.1, 0.2, 0.3]
    a = ARMA([0.3, -0.4, 0.4, -0.5, 0.6], [], 0.3)
    for mr in mrs:
        for i in range(20):
            time_series = [a.generater.next() for j in range(2000)]
            write(time_series, mr, i)

