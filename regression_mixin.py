

class RegressionMixin(object):

    def predict(self, X):
        return sum(map(lambda x, y: x*y,
                       self.avg_w, X))
