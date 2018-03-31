import numpy as np

from experiments.problems.multiObjective.structure.multiObjective import MultiObjective

class Sequence(MultiObjective):

    def __init__(self, sequences):
        self.chartype = np.dtype('|S1')
        self.empty = np.array([''], dtype=self.chartype)[0]

        self.sequences = np.array(sequences)

        self.lenths = np.array([len(r) for r in sequences])
        self.size = np.sum(self.lenths)
        self.maxlen = np.max(self.lenths)

        self.final_alignment = None

    def getFitness(self, position):
        index = 0
        shaped_position = []
        for hor in self.sequences:
            shaped_position.append(position[index:index+len(hor)])
            index += len(hor)

        mask = np.arange(self.maxlen) < self.lenths[:, None]
        padded_position = np.zeros(mask.shape)
        padded_position[mask] = np.concatenate(shaped_position)

        rounded_position = padded_position.astype(int)
        len_position = np.sum(rounded_position, axis=1)

        max_length = np.max(self.lenths + len_position)
        self.final_alignment = np.zeros((len(self.sequences), max_length), dtype=self.chartype)

        for i, seq in enumerate(self.sequences):
            index = 0
            for j in range(len(seq)):
                index += rounded_position[i][j]
                self.final_alignment[i][index] = seq[j]
                index += 1

        ra = np.rot90(self.final_alignment, -1)
        sum = 0
        valid = True
        for vert in ra:
            values, count = np.unique(vert, return_counts=True)
            if values[0] == self.empty:
                count = count[1:]
            if len(count) > 1:
                valid = False
                sum -= 1
            else:
                sum += np.sum(count - 1)

        return np.array([- sum, np.sum(padded_position)]), valid

    def printFinalAlignemnt(self):
        for hor in self.final_alignment:
            sequence = ''
            for vert in hor:
                isSpace = vert == self.empty
                if isSpace:
                    sequence += '_'
                else:
                    sequence += vert.decode('utf-8')
            print('\t' + sequence)