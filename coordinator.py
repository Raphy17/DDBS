from recPart import *

def enumerate_all_partitions():
    pass

def belongs_to(tuple, partitioning, dim, band_conditions): #returns the partition into which the tuple belong
    belongs_to = []

    if tuple[-1] == 0: #S tuple
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] <= tuple[d] < partitioning[i][d][1]):
                    is_part = False
            if is_part:
                belongs_to.append(i)
    else:
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] - band_conditions[d] <= tuple[d] < partitioning[i][d][1] + band_conditions[d]):
                    is_part = False
            if is_part:
                belongs_to.append(i)

    return belongs_to

def transform_recPart_into_partitioning(partitions):
    partitioning = []
    for p in partitions:
        if p.regular_partition:
            partitioning.append(tuple(p.A))
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
                    for j in range(i*(n/p.sub_partitions[d]), (i+1)*(n/p.sub_partitions[d])):
                        regions[j].append((p.A[d][0] + i * lengths[d]/p.sub_partitions[d], p.A[d][0] + (i+1) * lengths[d]/p.sub_partitions[d]))


    return partitioning




print(len(partitioning))
print(belongs_to(t, partitioning, 2, (5, 5)))