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

band_condition = [5, 5]
nr_w = 5
workers = []
for i in range(nr_w):
    worker = Worker(i)
    workers.append(worker)


S = []
T = []
k = 500
for w in workers:
    ts = w.get_sample("table_pareto15", k//nr_w)
    for t in ts:
        if t[-1] == 0:
            S.append(t)
        else:
            T.append(t)


S, T, parts, total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, over_head_history = recPart(S, T, band_condition, k, nr_w)


partitioning, loads = transform_recPart_into_partitioning(parts[-1])
p_to_w = distribute_partitions(partitioning, loads, nr_w)
#coordinator tells worker where to send their tuples
for w in workers:
    w.distribute_tuples("table_pareto15", workers, p_to_w, partitioning, 2, band_condition)

real_input = 0
output = []
for w in workers:
    real_input += len(w.tuples_to_join_T)
    real_input += len(w.tuples_to_join_S)
    output.extend(w.compute_output(band_condition))
print("-----")
print("Min input: " + str(k))
print("total input:" + str(total_input))
print("input overhead: " + str(overhead_input_dupl))
print("---load")
print("min workload per machine: " + str(l_zero))
print("workload of worst machine: " + str(l_max))
print("workload overhead: " + str(overhead_worker_load))
print("input before dupl:" + str(nr_w*3000))
print("real input:" + str(real_input))


print(len(output))
draw_partitions(S, T, parts)

