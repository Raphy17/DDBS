import time
from humanize import precisedelta
from datetime import timedelta

from DDBS.recPart import *
from DDBS.worker import *


def get_regions(regions, region, s_e, d):
    if d == len(s_e):
        regions.append(region)
        return
    elif d == 0:
        for r in s_e[0]:
            region = [r, ]
            get_regions(regions, region, s_e, d+1)
        return
    else:
        for r in s_e[d]:
            n_r = region.copy()
            n_r.append(r)
            get_regions(regions, n_r, s_e, d+1)


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
            start_endpoints = []
            lengths = []
            for d in p.A:
                lengths.append(d[1] - d[0])
                start_endpoints.append([])
            for d in range(len(lengths)):
                for i in range(p.sub_partitions[d]):
                    start_endpoints[d].append((p.A[d][0] + i * lengths[d]/p.sub_partitions[d], p.A[d][0] + (i+1) * lengths[d]/p.sub_partitions[d]))
            print(start_endpoints)

            regions = []
            get_regions(regions, [], start_endpoints, 0)

            for region in regions:
                if tuple(region) in partitioning:
                    print("KKK")
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

    start_time_rec_part = time.time()

    # Getting input sample for recPart algorithm from workers
    S, T = get_input_sample_from_workers(workers)

    best_partitioning, statistics = recPart(S, T, band_condition, k, nr_w)
    partitioning, loads = transform_recPart_into_partitioning(best_partitioning)
    p_to_w = distribute_partitions(loads, nr_w)

    end_time_rec_part = time.time()

    #visualizing, can be commented out (shows only first 2 dimensions)
    draw_partitions(S, T, statistics[0])

    start_total_time_distribution = time.time()
    for p in p_to_w.keys():
        workers[p_to_w[p]].initialize_tuples_to_join(p)

    #coordinator tells worker where to send their tuples
    single_time_distributions = []
    for w in workers:
        start_single_time_distribution = time.time()
        w.distribute_tuples("table_pareto15", workers, p_to_w, partitioning, len(band_condition), band_condition)
        end_single_time_distribution = time.time()
        single_time_distributions.append(end_single_time_distribution-start_single_time_distribution)

    slowest_single_time_distribution = max(single_time_distributions)

    end_total_time_distribution = time.time()

    start_total_time_join = time.time()
    output = []
    single_time_join = []
    for w in workers:
        start_single_time_join = time.time()
        output.extend(w.compute_output(band_condition))
        end_single_time_join = time.time()
        single_time_join.append(end_single_time_join - start_single_time_join)

    end_total_time_join = time.time()
    slowest_single_time_join = max(single_time_join)

    #CALCULATRE/SHOW SOME TIME STATISTICS
    print("Duration rec part: ", precisedelta(timedelta(seconds=end_time_rec_part - start_time_rec_part)))
    print("Total time distribution: ", precisedelta(timedelta(seconds=end_total_time_distribution - start_total_time_distribution)))
    print("Slowest time single distribution: ", precisedelta(timedelta(seconds=slowest_single_time_distribution)))
    print("Total time join: ", precisedelta(timedelta(seconds=end_total_time_join - start_total_time_join)))
    print("Slowest time single distribution: ", precisedelta(timedelta(seconds=slowest_single_time_join)))

    #GETTING SOME STATISTICS
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


