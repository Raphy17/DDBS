from coordinator import coordinate_join

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
