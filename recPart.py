import random

def per_worker_load_Variance(partitions, w):    #uses for beta 2, beta 4: (4, 1) (like in amazon cloud cluster)
    Vp = (w - 1) / w**2
    tmp = 0
    for p in partitions:
        load_of_p = 4 * p.get_input_size() + 1 * p.get_output_size()
        tmp += load_of_p ** 2
    Vp *= tmp
    return Vp

class Partition():          #tuple structure: the join necessary dimensions at the front and table/sample membership(either S/T) at the end e.g. join on age, loc_x, loc_y --> tuple (age, lox_x, loc_y, name, bla, bla, S/T)

    def __init__(self, A, sample_S, sample_T, sample_output):
        self.__dim = dim
        self.sample_S = sample_S
        self.sample_T = sample_T
        self.sample_input = sample_S.extend(sample_T)
        self.sample_output = sample_output      #gets calculated
        self.__input_size = len(self.sample_input)
        self.__output_size = len(sample_output)
        self.bestSplit, self.topScore = self.bestSplit()
        self.A = A      #e.g. A = [(20, 30), (1031, 1300), (742, 935)] age 20-30, x_loc 1031-1300, y_loc 742-935

    def best_split(self, partitions, w):
        bestSplit = None
        topScore = 0
        Vp = per_worker_load_Variance(partitions, w) #before applying partitioning
        for dim in range(self.__dim):       #find best split out of all dimensions
            best_x = 0
            score_best_x = 0
            self.sample_input.sort(key=lambda x:x[dim])    #sort input_sample on dimension A
            for i in range(0, len(self.sample_input)-1):    #find best split a single dimension
                x = (self.sample_input[i] + self.sample_input[i+1])/2
                delta_var_x = 4
                delta_dup_x = 2
                score_x = delta_var_x/delta_dup_x
                if score_x > score_best_x:
                    score_best_x = score_best_x
                    best_x = x
            if score_best_x > topScore:
                topScore = score_best_x
                bestSplit = best_x


        return (bestSplit, topScore)

    def apply_best_split(self):
        return

    def get_input_size(self):
        return self.__input_size

    def get_output_size(self):
        return self.__output_size


def compute_output(S, T, condition):
    pass


def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(100):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample


def split_score(partitions, w, pa): #w = number of workers, p = partition,




def recPart(dim, S, T, condition, k):               #condition = epsilon for each band-join-dimension e.g. (10, 100, 100) for 10 years apart, 100km ind x and y direction
    random_sample_S = draw_random_sample(S, k/2, 0)
    random_sample_T = draw_random_sample(T, k/2, 1)
    random_output_sample = compute_output(random_sample_S, random_sample_T, condition)
    partitions = []         #all partitions
    A = [(0, 100), (0, 1000), (0, 1000)]  #because our random samples have values in between
    root_p = Partition(A, random_sample_S, random_sample_T, random_output_sample)
    partitions.append(root_p)
    pass
