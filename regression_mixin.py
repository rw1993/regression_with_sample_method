

class RegressionMixin(object):

    def predict(self, X):
        return sum(map(lambda x, y: x*y,
                       self.avg_w, X))
    @property
    def avg_w(self):
        avg = [.0 for i in range(self.w_dim)]
        for w in self.ws:
            avg = map(lambda x, y: x+y,
                           avg, w)
        avg = [w/len(self.ws) for w in avg]
        return avg

