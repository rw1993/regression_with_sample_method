

class RegressionMixin(object):

    def re_init(self):
        self._avg_w = None

    def predict(self, X):
        return sum(map(lambda x, y: x*y,
                       self.avg_w, X))
    def predict_label(self, X):
        if self.predict(X) > 0:
            return 1
        else:
            return -1
    @property
    def avg_w(self):
        if not self._avg_w:
            avg = [.0 for i in range(self.w_dim)]
            for w in self.ws:
                avg = map(lambda x, y: x+y,
                               avg, w)
            self._avg_w = [w/len(self.ws) for w in avg]
        return self._avg_w

