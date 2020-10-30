import random


def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(100):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample

def compute_max_worker_load(partitions, w):
    worker_loads = [0,]*w
    partition_loads = partitions

    partition_loads.sort(reverse=True)
    for load in partition_loads:
        worker_loads[worker_loads.index(min(worker_loads))] += load

    return max(worker_loads)

p = [1, 2, 2, 4, 4, 5, 5, 7, 10]
print(compute_max_worker_load(p, 4))



