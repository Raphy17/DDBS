from dbs import *

def belongs_to(tuple, partitioning, dim, band_conditions): #returns the partition into which the tuple belong
    belongs_to = []

    if tuple[-1] == 0: #S tuple
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] <= tuple[d] <= partitioning[i][d][1]):
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
        self.tuples_to_join_S = []
        self.tuples_to_join_T = []


    def get_sample(self, table, k):
        samples = self.db.get_k_random_samples(table, k)
        return samples

    def distribute_tuples(self, table, workers, p_to_w, partitioning, dim, band_condition):
        all_t = self.db.get_table(table)

        for t in all_t:
            part_of_partitions = belongs_to(t, partitioning, dim, band_condition)
            for p in part_of_partitions:
                self.send_to(workers[p_to_w[p]], t)

    def send_to(self, w, t):
        w.receive(t)

    def receive(self, t):
        if t[-1] == 0:
            self.tuples_to_join_S.append(t)
        else:
            self.tuples_to_join_T.append(t)

    def compute_output(self, band_conditions):
        S = self.tuples_to_join_S
        T = self.tuples_to_join_T
        output = []
        for s_element in S:
            for t_element in T:
                joins = True
                for i in range(len(band_conditions)):
                    if not (abs(s_element[i] - t_element[i]) <= band_conditions[i]):
                        joins = False
                if joins:
                    output.append((s_element, t_element))
        return output

