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


def create_workers(nr_w):
    workers = []
    for i in range(nr_w):
        worker = Worker(i)
        workers.append(worker)
    return workers


def get_input_sample_from_workers(workers):
    S = []
    T = []
    for w in workers:
        ts = w.get_sample("table_pareto15", k // nr_w)
        for t in ts:
            if t[-1] == 0:
                S.append(t)
            else:
                T.append(t)
    return S, T


def distribute_partitions(loads, w):
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


if __name__ == '__main__':

    band_condition = [2, 2, 2]     #band join predicate
    nr_w = 5                  #number of workers
    k = 100                     #sample size (best to choose something divisible by nr_w)

    #creating a worker for each Database
    workers = create_workers(nr_w)

    # Getting input sample for recPart algorithm from workers
    S, T = get_input_sample_from_workers(workers)

    best_partitioning, statistics = recPart(S, T, band_condition, k, nr_w)

    partitioning, loads = transform_recPart_into_partitioning(best_partitioning)
    p_to_w = distribute_partitions(loads, nr_w)

    for p in p_to_w.keys():
        workers[p_to_w[p]].initialize_tuples_to_join(p)

    (print("parts"))
    print(p_to_w)
    print(loads)
    print(partitioning)
    draw_partitions(S, T, statistics[0])

    #coordinator tells worker where to send their tuples


    for w in workers:
        w.distribute_tuples("table_pareto15", workers, p_to_w, partitioning, len(band_condition), band_condition)

    #start total time here
    output = []
    for w in workers:
        output.extend(w.compute_output(band_condition))
    #end total time here
    #calculate time of slowest worker here


    parts, total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, over_head_history = statistics
    #calculates duplication
    real_input = 0
    for w in workers:
        for key in w.tuples_to_join_S:
            real_input += len(w.tuples_to_join_S[key])
            real_input += len(w.tuples_to_join_T[key])


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


    #draws only the first 2 dimensions of the partitions, mighjt look strange


