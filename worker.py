from dbs import *

def belongs_to(tuple, partitioning, dim, band_conditions): #returns the partition into which the tuple belong
    belongs_to = []

    if tuple[-1] == 0: #S tuple
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] <= tuple[d] < partitioning[i][d][1]):
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

    return belongs_to

class Worker():
    counter_w = 0
    def __init__(self, nr):
        self.db = self.create_db(str(nr) + "dbs.db")
        self.counter_w += 1
        self.tuples_to_join = []

    def create_db(self, path):
        db = Database(path)
        db.create_table("table_un")
        db.fill_table("table_un", 1, 100)
        return db

    def get_sample(self, k):
        samples = self.db.get_k_random_samples("table_un", k)
        return samples

    def distribute_tuples(self, workers, p_to_w, partitioning, dim, band_condition):
        all_t = self.db.get_table("table_un")

        for t in all_t:
            part_of_partitions = belongs_to(t, partitioning, dim, band_condition)
            for p in part_of_partitions:
                self.send_to(workers[p_to_w[p]], t)

    def send_to(self, w, t):
        w.receive(t)

    def receive(self, t):
        self.tuples_to_join.append(t)


