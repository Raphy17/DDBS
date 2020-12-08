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

            regions = []
            get_regions(regions, [], start_endpoints, 0)

            for region in regions:
                if tuple(region) in partitioning:
                    print("KKK")
                partitioning.append(tuple(region))
                loads.append(p.get_load()[0])
    return partitioning, loads


def create_workers(nr_w, size):
    workers = []
    for i in range(nr_w):
        worker = Worker(i, size)
        workers.append(worker)
    return workers


def get_input_sample_from_workers(workers):
    S = []
    T = []
    for w in workers:
        ts = w.get_sample("table_pareto15", sample_size // nr_w)
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

    for p in parts:
        worker = worker_loads.index(min(worker_loads))
        worker_loads[worker] += p[1]
        partition_to_worker[p[0]] = worker
    return partition_to_worker

def coordinate_join(band_condition, nr_w, sample_size, size):

    #creating a worker for each Database
    workers = create_workers(nr_w, size)

    start_time_rec_part = time.time()

    # Getting input sample for recPart algorithm from workers
    S, T = get_input_sample_from_workers(workers)

    best_partitioning, recPart_statistics = recPart(S, T, band_condition, sample_size, nr_w)
    partitioning, loads = transform_recPart_into_partitioning(best_partitioning)
    p_to_w = distribute_partitions(loads, nr_w)

    end_time_rec_part = time.time()

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

    #calculating some statistics
    parts, total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, over_head_history = recPart_statistics
    real_input = 0
    real_loads = []
    for w in workers:
        real_input += w.get_join_input_size()
        real_loads.append(w.get_load())
    min_real_workload_per_machine = (4*nr_w*size + len(output))/nr_w
    workload_of_worst_machine = max(real_loads)

    #recPart, total Distribution, slowest distribution, total join, slowest join
    time_statistics = []
    time_statistics.append(end_time_rec_part - start_time_rec_part)
    time_statistics.append(end_total_time_distribution - start_total_time_distribution)
    time_statistics.append(slowest_single_time_distribution)
    time_statistics.append(end_total_time_join - start_total_time_join)
    time_statistics.append(slowest_single_time_join)

    #
    dupl_statistics = []
    dupl_statistics.append(sample_size)
    dupl_statistics.append(total_input)
    dupl_statistics.append(overhead_input_dupl)
    dupl_statistics.append(nr_w*size)
    dupl_statistics.append(nr_w*size*(overhead_input_dupl+1))
    dupl_statistics.append(real_input)
    dupl_statistics.append((real_input-nr_w*size)/(nr_w*size))
    dupl_statistics.append(len(output))

    #
    load_statistics = []
    load_statistics.append(l_zero)
    load_statistics.append(l_max)
    load_statistics.append(overhead_worker_load)
    load_statistics.append(min_real_workload_per_machine)
    load_statistics.append(workload_of_worst_machine)
    load_statistics.append((workload_of_worst_machine-min_real_workload_per_machine)/min_real_workload_per_machine)
    statistics = (time_statistics, dupl_statistics, load_statistics)

    return output, statistics





if __name__ == '__main__':

    band_condition = [2, 2, 2]      #band join predicate
    nr_w = 5                        #number of workers
    sample_size = 500               #sample size (best to choose something divisible by nr_w)
    size = 2000                     #size of tables per Database (dbs are filled up to 10'000 at the moment)

    output, statistics = coordinate_join(band_condition, nr_w, sample_size, size)


    #CALCULATRE/SHOW SOME TIME STATISTICS
    print("---time statistics")
    print("Duration rec part: ", precisedelta(timedelta(seconds=statistics[0][0])))
    print("Total time distribution: ", precisedelta(timedelta(seconds=statistics[0][1])))
    print("Slowest time single distribution: ", precisedelta(timedelta(seconds=statistics[0][2])))
    print("Total time join: ", precisedelta(timedelta(seconds=statistics[0][3])))
    print("Slowest time single join: ", precisedelta(timedelta(seconds=statistics[0][4])))

    print("---duplication statistics")
    print("Sample input before duplication: " + str(statistics[1][0]))
    print("Total sample input after duplication:" + str(statistics[1][1]))
    print("Estimated Input overhead: " + str(statistics[1][2]))
    print("Real Input before duplication:" + str(statistics[1][3]))
    print("Estimated Input after duplication:" + str(statistics[1][4]))
    print("Real Input after duplication:" + str(statistics[1][5]))
    print("Real Input overhead: " + str(statistics[1][6]))
    print("Output size:" + str(statistics[1][7]))

    print("---load statistics")
    print("lower bound workload per machine: " + str(statistics[2][0]))
    print("workload of worst machine: " + str(statistics[2][1]))
    print("Estimated workload overhead: " + str(statistics[2][2]))
    print("real lower bound workload per machine: " + str(statistics[2][3]))
    print("real workload of worst machine: " + str(statistics[2][4]))
    print("real workload overhead: " + str(statistics[2][5]))


    #draws only the first 2 dimensions of the partitions, mighjt look strange


