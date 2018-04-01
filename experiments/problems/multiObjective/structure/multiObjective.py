
class MultiObjective(object):

    def __init__(self, sequences):
        raise Exception('__init__ not implemented.')

    def getFitness(self, position):
        raise Exception('Get fitness not implemented.')

    def dominant(self, x1, x2):
        return all(x1 <= x2) and any(x1 < x2)