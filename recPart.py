import random

def load(input_size, output_size, b2, b3):
    return b2*input_size+b3*output_size


def per_worker_load_Variance(partitions, w):    #uses for beta 2, beta 3: (4, 1) (like in amazon cloud cluster)
    Vp = (w - 1) / w**2
    tmp = 0
    for p in partitions:
        load_of_p = load(p.get_input_size(), p.get_output_size(), 1, 4)
        tmp += load_of_p ** 2
    Vp *= tmp
    return Vp


def find_dupl(a, i, eps, dim):
    dupl = 0
    v_i_plus_1 = a[i+1][dim]
    v_i = a[i][dim]
    j = i
    while j >=0:
        if v_i_plus_1 - a[j][dim] > eps:
            break
        j -= 1
        dupl += 1
    j = i+1
    while j <= len(a) -1:
        if a[j][dim] - v_i > eps:
            break
        j += 1
        dupl += 1
    return dupl


class Partition():          #tuple structure: the join necessary dimensions at the front and table/sample membership(either S/T) at the end e.g. join on age, loc_x, loc_y --> tuple (age, lox_x, loc_y, name, bla, bla, S/T)

    def __init__(self, A, sample_S, sample_T, sample_output):
        self.sample_S = sample_S
        self.sample_T = sample_T
        self.sample_input = sample_S.copy() + sample_T.copy()
        self.sample_output = sample_output      #gets calculated
        self.A = A      #e.g. A = [(20, 30), (1031, 1300), (742, 935)] age 20-30, x_loc 1031-1300, y_loc 742-935
        self.top_score = None
        self.best_split = None
        self.dim_best_split = None

    def find_best_split(self, partitions, w):
        best_split = None
        top_score = 0
        dim_best_split = 0
        Vp = per_worker_load_Variance(partitions, w) #before applying partitioning
        for dim in range(len(self.A)):       #find best split out of all dimensions
            best_x = 0
            score_best_x = 0
            self.sample_input.sort(key=lambda x:x[dim])    #sort input_sample on dimension A
            for i in range(0, len(self.sample_input)-1):    #find best split a single dimension
                x = (self.sample_input[i][dim] + self.sample_input[i+1][dim])/2
                delta_dup_x = find_dupl(self.sample_input, i, 5, dim)  #replace 5 with conditio
                Vp_new = Vp - (w-1)/w**2 * (load(self.get_input_size(), self.get_output_size(), 4, 1)**2)
                Vp_new += (w-1)/w**2 * (load(1+i+delta_dup_x, 1+i+delta_dup_x, 4, 1)**2 + load(len(self.sample_input)-1-i+delta_dup_x, len(self.sample_input)-1-i+delta_dup_x, 4, 1)**2)
                delta_var_x = Vp_new - Vp
                if delta_dup_x == 0:
                    delta_dup_x = 1
                score_x = delta_var_x/delta_dup_x
                if score_x > score_best_x:
                    score_best_x = score_best_x
                    best_x = x
            if score_best_x > top_score:
                top_score = score_best_x
                best_split = best_x
                dim_best_split = dim
        self.top_score = top_score
        self.best_split = None
        self.dim_best_split = None

        return best_split, top_score, dim_best_split


    def apply_best_split(self):
        p_new_1 = 2     #return newly formed partitions
        p_new_2 = 1
        return p_new_1, p_new_2

    def get_input_size(self):
        return len(self.sample_input)

    def get_output_size(self):
        return len(self.sample_output)

    def get_topScore(self):
        return self.top_score

    def get_best_split(self):
        return self.best_split

def compute_output(S, T, condition):
    return S


def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(k):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample


def find_top_score_partition(partitions):
    top_score_partition = None
    score = 0
    for p in partitions:
        if p.get_topScore() > score:
            score = p.get_topScore()
            top_score_partition = p
    return p


def recPart(dim, S, T, condition, k, w):               #condition = epsilon for each band-join-dimension e.g. (10, 100, 100) for 10 years apart, 100km ind x and y direction
    random_sample_S = draw_random_sample(S, k//2, 0)
    random_sample_T = draw_random_sample(T, k//2, 1)
    random_output_sample = compute_output(random_sample_S, random_sample_T, condition)
    partitions = []         #all partitions
    A = [(0, 100), (0, 1000), (0, 1000)]  #because our random samples have values in between these domains
    root_p = Partition(A, random_sample_S, random_sample_T, random_output_sample)
    partitions.append(root_p)
    root_p.find_best_split(partitions, w)

    termination_condition = True
    while termination_condition:
        p_max = find_top_score_partition(partitions)
        p_new_1, p_new_2 = p_max.apply_best_split()
        p_new_1.find_best_split(partitions, w)
        p_new_2.find_best_split(partitions, w)

    return partitions

recPart(3, 2, 2, 3, 10, 10)
