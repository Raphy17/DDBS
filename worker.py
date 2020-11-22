from dbs import *

def belongs_to(tuple, partitioning, dim, band_conditions): #returns the partition into which the tuple belong
    belongs_to = []

    if tuple[-1] == 0: #S tuple
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                lower_b = partitioning[i][d][0]
                upper_b = partitioning[i][d][1]
                if lower_b == 1:        #needs to be changed for normal distributions (works for domain : [1, infinity])
                    lower_b = 0.999
                if not (lower_b < tuple[d] <= upper_b):
                    is_part = False
            if is_part:
                belongs_to.append(i)
    else:
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] - band_conditions[d] <= tuple[d] < partitioning[i][d][1] + band_conditions[d]):
                    is_part = False
            if is_part:
                belongs_to.append(i)
    if len(belongs_to) > 1 and tuple[-1] == 0:
        print("!!!")
    return belongs_to

class Worker():
    counter_w = 0
    def __init__(self, nr):
        self.db = Database(str(nr) + "dbs.db")
        self.counter_w += 1
        self.tuples_to_join_S = {}
        self.tuples_to_join_T = {}


    def get_sample(self, table, k):
        samples = self.db.get_k_random_samples(table, k)
        return samples

    def initialize_tuples_to_join(self, p):
        self.tuples_to_join_S[p] = []
        self.tuples_to_join_T[p] = []

    def distribute_tuples(self, table, workers, p_to_w, partitioning, dim, band_condition):
        all_t = self.db.get_table(table)

        for t in all_t:
            part_of_partitions = belongs_to(t, partitioning, dim, band_condition)
            for p in part_of_partitions:
                self.send_to(workers[p_to_w[p]], p, t)

    def send_to(self, w, p, t):
        w.receive(p, t)

    def receive(self, p, t):
        if t[-1] == 0:
            self.tuples_to_join_S[p].append(t)
        else:
            self.tuples_to_join_T[p].append(t)

    def compute_output(self, band_conditions):
        output = []
        for partition in self.tuples_to_join_S.keys():
            S = self.tuples_to_join_S[partition]
            T = self.tuples_to_join_T[partition]
            for s_element in S:
                for t_element in T:
                    joins = True
                    for i in range(len(band_conditions)):
                        if not (abs(s_element[i] - t_element[i]) <= band_conditions[i]):
                            joins = False
                    if joins:
                        output.append((s_element, t_element))
        return output

