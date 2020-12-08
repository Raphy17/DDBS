from coordinator import coordinate_join
import csv
#band_width
#dim 1 -5
#input size 1000-20000
#workers


for input_size in range(1000, 20000, 200):
    band_condition = [2, 2, 2]      #band join predicate
    nr_w = 5                        #number of workers
    sample_size = input_size//20               #sample size (best to choose something divisible by nr_w)
    size = input_size//nr_w                     #size of table per Database (first 5 dbs are fille up to 20'000 at the moment, last 5 up to 10'000)
    all_statistics = []
    output, statistics = coordinate_join(band_condition, nr_w, sample_size, size)
    all_statistics.append(statistics)


def write_csv_file(file, all_statistics):
    with open(file, "r+") as f:
        writer = csv.writer(file)
        writer.writerow(["Duration rec part", "Total time distribution", "Slowest time single distribution", "Total time join",
                  "Slowest time single join", "Total time", "Percentage of time spent in recPart",
                  "Sample input before duplication", "Total sample input after duplication", "Estimated Input overhead",
                  "Real Input before duplication", "Estimated Input after duplication", "Real Input after duplication",
                  "Real Input overhead", "Output size",
                  "lower bound workload per machine", "workload of worst machine", "Estimated workload overhead",
                  "real lower bound workload per machine", "real workload of worst machine", "real workload overhead"
                  ])



        for statistics in all_statistics:
            a, b, c = statistics
            range = len(a)+len(b)+len(c)

            for i in range(range):
                writer.writerow([a[0], a[1], a[2], a[3], a[4], a[5], a[6],
                                 b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
                                 c[0], c[1], c[2], c[3], c[4], c[5]])


