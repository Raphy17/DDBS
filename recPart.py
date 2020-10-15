

class Partition():

    def __init__(self, dim, sample_S, sample_T, sample_output):
        self.__dim = dim
        self.sample_S = sample_S
        self.sample_T = sample_T
        self.sample_output = sample_output
        self.bestSplit, self.topScore = self.bestSplit()
        A = [max()]

    def best_split(self):
        bestSplit = 5
        topScore = 100
        return (bestSplit, topScore)

    def apply_best_split(self):
        return


def compute_output(S, T, condition):
    pass


def draw_random_sample(R, k):
    pass


def recPart(dim, S, T, condition, k):
    random_sample_S = draw_random_sample(S, k/2)
    random_sample_T = draw_random_sample(T, k/2)
    random_output_sample = compute_output(random_sample_S, random_sample_T, condition)
    root_p = Partition(dim)