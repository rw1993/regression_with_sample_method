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
