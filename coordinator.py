from recPart import *
from worker import *



def transform_recPart_into_partitioning(partitions):
    partitioning = []
    loads = []
    for p in partitions:
        if p.regular_partition:
            part = []
            for d in range(len(p.A)):
                part.append((p.A[d][0], p.A[d][1]))
            partitioning.append(tuple(part))
            loads.append(p.get_load()[0])
        else:
            lengths = []
            for d in p.A:
                lengths.append(d[1] - d[0])
            regions = []
            n = p.nr_of_regions
            for i in range(n):
                regions.append([])
            for d in range(len(lengths)):
                for i in range(p.sub_partitions[d]):

                    for j in range(i*(n//p.sub_partitions[d]), (i+1)*(n//p.sub_partitions[d])):
                        regions[j].append((p.A[d][0] + i * lengths[d]/p.sub_partitions[d], p.A[d][0] + (i+1) * lengths[d]/p.sub_partitions[d]))
            for region in regions:

                partitioning.append(tuple(region))
                loads.append(p.get_load()[0])
    return partitioning, loads


def distribute_partitions(partitioning, loads, w):
    worker_loads = [0, ] * w
    partition_to_worker = {}
    parts = [(i, loads[i]) for i in range(len(loads))]
    parts.sort(key=lambda x: x[1], reverse=True)
    print(parts)
    for p in parts:
        worker = worker_loads.index(min(worker_loads))
        worker_loads[worker] += p[1]
        partition_to_worker[p[0]] = worker
    return partition_to_worker


s, t, parts, total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, over_head_history = recPart(1, 2, [5, 5], 1000, 3)
print(parts[-1])
draw_partitions(s, t, parts)
partitioning, loads =transform_recPart_into_partitioning(parts[-1])
p_to_w = distribute_partitions(partitioning, loads, 3)
worker_0 = Worker(0)
worker_1 = Worker(1)
worker_2 = Worker(2)
workers = [worker_0, worker_1, worker_2]
for w in workers:
    w.distribute_tuples(workers, p_to_w, partitioning, 2, [5, 5])

print(worker_0.tuples_to_join)
print(len(worker_0.tuples_to_join))
print(worker_1.tuples_to_join)
print(len(worker_1.tuples_to_join))
print(worker_2.tuples_to_join)
print(len(worker_2.tuples_to_join))
