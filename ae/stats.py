
class RunningMeanAndVar:

    def __init__(self):
        self.existingAggregate = (0, 0, 0)

    def add(self, newValue):
        self.existingAggregate = RunningMeanAndVar.update(self.existingAggregate, newValue)

    def values(self):
        tmp = RunningMeanAndVar.finalize(self.existingAggregate)
        return {'mean': tmp[0], 'var': tmp[1]}

    # Taken from Wikipedia article https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    #
    # For a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    @staticmethod
    def update(existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2
        return (count, mean, M2)

    # Retrieve the mean, variance and sample variance from an aggregate
    @staticmethod
    def finalize(existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float("nan")
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
            return (mean, variance, sampleVariance)