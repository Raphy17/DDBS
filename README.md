# DDBS

Implementation of the distributed band-join algorithm descibred in the following paper:

Li, Rundong, et al. “Near-Optimal Distributed Band-Joins through Recursive Partitioning.” *Proceedings of the 2020 ACM SIGMOD International Conference on Management of Data*, ACM, 2020, pp. 2375–90. doi:10.1145/3318464.3389750.

## Setup
It is recommended that you install required packages in a new virtulenv from `requirements.txt`. Requirements are built from `requirements.in` using `pip-tools`

## Running code
There are 3 main files you can run

recPart: Let this run if you want to compute the partitioning

Computes a partitioning based on the recPart algorithm by Li, Rundong, et al.
You can adjust number of workers, sample_size, band_condition and distribution being used.

coordinator: Let this run if you want to compute a full-join

First collects an input sample from the DBS's, then uses recPart to find best partitioning and finally computes the full band-join based on that partitioning.
You can adjust number of workers, join_siz, sample_size, band_condition and distribution being used. 
The maximum size for the first 5 workers is 20'000 tuples and for the rest 10'000, if you desire to let bigger joins run, the DBS's first need to be filled more.



Tests: Let this run if you want to test coordinator or recPart on a range of different inputs and automatically 
protocol the results in an excel file.