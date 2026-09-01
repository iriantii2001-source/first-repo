class AverageValueMeter(object):
    def __init__(self):
        self.reset()

    def add(self, value):
        self.sum += value
        self.sum_sq += value * value
        self.n += 1

    def value(self):
        if self.n == 0:
            return float('nan'), float('nan')

        mean = self.sum / self.n
        if self.n > 1:
            var = (self.sum_sq - self.n * mean * mean) / (self.n - 1)
            std = var ** 0.5 if var > 0 else 0.0
        else:
            std = float('nan')

        return mean, std

    def reset(self):
        self.sum = 0.0
        self.sum_sq = 0.0
        self.n = 0
