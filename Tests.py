from coordinator import coordinate_join
from recPart import *
import csv
import pandas as pd
import time
#band_width
#dim 1 -5
#input size 1000-20000
#workers

# all_statistics = []
# for input_size in range(1000, 20000, 200):
#     band_condition = [2, 2, 2]      #band join predicate
#     nr_w = 5                        #number of workers
#     sample_size = input_size//20               #sample size (best to choose something divisible by nr_w)
#     size = input_size//nr_w                     #size of table per Database (first 5 dbs are fille up to 20'000 at the moment, last 5 up to 10'000)
#     output, statistics = coordinate_join(band_condition, nr_w, sample_size, size)
#     all_statistics.append(statistics)
#     print(input_size)

all_statistics = []
for sample_size in range(100, 10000, 100):
    w = 5                       #nr of workers
    band_condition = [2, 2, 2]     #band condition
    Time = 0
    dupl_overhead = 0
    workload_overhead =  0
    for i in range(1):
        random_sample_S = construct_pareto_data(sample_size // 2, 1.5, 0, len(band_condition))
        random_sample_T = construct_pareto_data(sample_size // 2, 1.5, 1, len(band_condition))
        start_time_rec_part = time.time()
        best_partitioning, statistics = recPart(random_sample_S, random_sample_T, band_condition, sample_size, w)
        end_time_rec_part = time.time()
        Time += end_time_rec_part-start_time_rec_part
        dupl_overhead += statistics[3]
        workload_overhead += statistics[4]
    all_statistics.append((sample_size, Time, dupl_overhead, workload_overhead))
    print(sample_size)
    print(all_statistics)

# def write_csv_file(file, results):
#     with open(file, "w") as f:
#         writer = csv.writer(f)
#         writer.writerow(["Duration rec part", "Total time distribution", "Slowest time single distribution",
#                         "Total time join", "Slowest time single join", "Total time",
#                         "Percentage of time spent in recPart",
#                         "Sample input before duplication", "Total sample input after duplication",
#                         "Estimated Input overhead", "Real Input before duplication", "Estimated Input after duplication",
#                         "Real Input after duplication", "Real Input overhead", "Output size",
#                         "lower bound workload per machine", "workload of worst machine", "Estimated workload overhead",
#                         "real lower bound workload per machine", "real workload of worst machine", "real workload overhead"
#                   ])
#
#         for stat in results:
#             a, b, c = stat
#
#             writer.writerow([a[0], a[1], a[2], a[3], a[4], a[5], a[6],
#                              b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
#                              c[0], c[1], c[2], c[3], c[4], c[5]
#                             ])

def write_csv_file(file, results):
    with open(file, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["Sample size", "Time", "Duplication overhead", "Workload overhead"])

        for stat in results:
            writer.writerow([stat[0], stat[1], stat[2], stat[3]])



write_csv_file("Results.csv", all_statistics)
#make excel file
read_file = pd.read_csv(r'Results.csv')
read_file.to_excel(r'Results.xlsx', index=None, header=True)
