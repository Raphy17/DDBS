#recPart but duplication and load variance are not extrapolated

import math
import random
import pandas as pd
from math import pi
import numpy as np
from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Line, HoverTool, FactorRange
import bokeh.palettes as bp


def load(input_size, output_size, b2, b3):
    return (b2 * input_size + b3 * output_size)         #1:1000 = ratio sample:real


def per_worker_load_variance(partitions, w):  # uses for beta 2, beta 3: (4, 1) (like in amazon cloud cluster)
    Vp = (w - 1) / w ** 2
    tmp = 0
    for p in partitions:
        load_of_p = p.get_load()
        for load_sub_partition in load_of_p:
            tmp += load_sub_partition** 2
    Vp *= tmp
    return Vp


def find_dupl(a, i, band, dim):
    dupl = 0
    for j in range(i + 1, len(a)):
        v_i_plus_1 = a[j][dim]
        if a[j][-1] == 0:
            break
    for j in range(i, -1, -1):
        v_i = a[j][dim]
        if a[j][-1] == 0:
            break
    j = i
    while j >= 0:
        if v_i_plus_1 - a[j][dim] > band:
            break

        if a[j][-1] == 1:  # 1 tuple belongs to sample T, 0 tuple belong to sample S
            dupl += 1
        j -= 1

    j = i + 1
    while j <= len(a) - 1:
        if a[j][dim] - v_i > band:
            break
        if a[j][-1] == 1:  # 1 tuple belongs to sample T, 0 tuple belong to sample S
            dupl += 1
        j += 1

    return dupl


class Partition:  # tuple structure: the join necessary dimensions at the front and table/sample membership(either
    # S/T) at the end e.g. join on age, loc_x, loc_y --> tuple (age, lox_x, loc_y, name, bla, bla, S/T)

    def __init__(self, A, sample_S, sample_T, sample_output):
        self.sample_S = sample_S
        self.sample_T = sample_T
        self.sample_input = sample_S.copy() + sample_T.copy()
        self.sample_output = sample_output  # gets calculated
        self.A = A  # e.g. A = [(20, 30), (1031, 1300), (742, 935)] age 20-30, x_loc 1031-1300, y_loc 742-935
        self.top_score = 0
        self.best_split = None
        self.dim_best_split = None
        self.delta_dupl_best_split = 0
        self.dupl_caused = 0
        self.sub_partitions = len(A)*[1,]
        self.regular_partition = True
        self.nr_of_regions = 1          #number of small partition regions

    def __repr__(self):
        if self.regular_partition:
            return "{}".format(self.A)
        else:
            return "{}sub:{}".format(self.A, self.sub_partitions)

    def find_best_split(self, partitions, band_condition, w):
        valid_dims = self.get_valid_dims(band_condition)
        best_split = 0
        top_score = 0
        dim_best_split = 0
        dupl_best_split = 0
        if len(valid_dims) > 0:  # if len == 0, its a small partition -> use 1 bucket
            Vp = per_worker_load_variance(partitions, w)  # before applying partitioning
            for dim in valid_dims:  # find best split out of all dimensions
                best_x = 0
                score_best_x = 0
                dupl_best_x = 0
                self.sample_input.sort(key=lambda x: x[dim])  # sort input_sample on dimension A
                for i in range(0, len(self.sample_input) - 1):  # find best split a single dimension
                    x = (self.sample_input[i][dim] + self.sample_input[i + 1][dim]) / 2
                    if not (self.A[dim][0] + band_condition[dim]/3 < x < self.A[dim][1] - band_condition[dim]/3):
                        continue
                    delta_dup_x = find_dupl(self.sample_input, i, band_condition[dim], dim)
                    Vp_new = Vp - (w - 1) / w ** 2 * (self.get_load()[0] ** 2)
                    Vp_new += (w - 1) / w ** 2 * (load(1 + i + delta_dup_x, (self.get_output_size()/self.get_input_size())*(1 + i), 4, 1) ** 2 + load(
                        len(self.sample_input) - 1 - i + delta_dup_x, (self.get_output_size()/self.get_input_size())*(len(self.sample_input) - 1 - i), 4,
                        1) ** 2) #removed duplication from adding to variance, since it's ambiguous in paper
                    delta_var_x = Vp - Vp_new
                    if delta_dup_x == 0:
                        delta_dup_x = 1
                    score_x = delta_var_x / delta_dup_x
                    if score_x > score_best_x:
                        score_best_x = score_x
                        best_x = x
                        dupl_best_x = delta_dup_x
                if score_best_x > top_score:
                    top_score = score_best_x
                    best_split = best_x
                    dim_best_split = dim
                    dupl_best_split = dupl_best_x
            self.top_score = top_score
            self.best_split = best_split
            self.dim_best_split = dim_best_split
            self.delta_dupl_best_split = dupl_best_split
            if self.top_score > 0:
                return self.best_split, self.top_score, self.dim_best_split, self.delta_dupl_best_split

        #1 bucket if dimensions are to small or no regular topscore has value > 0
        delta_Dupl = self.duplication_caused_by_small_partitioning()  #array with the delta_dupl causes when increasing number of splits in a dimension by 1 (e.g. increasing rows from 1 to 2
        delta_Var = self.delta_Var_caused_by_small_partitioning()
        sigma_dim = []
        for i in range(len(delta_Var)):
            if delta_Dupl[i] == 0:
                delta_Dupl = 1
            sigma_dim.append(delta_Var[i]/delta_Dupl[i])

        self.top_score = max(sigma_dim)

        self.dim_best_split = sigma_dim.index(self.top_score)
        self.best_split = self.dim_best_split
        self.regular_partition = False
        self.delta_dupl_best_split = delta_Dupl[self.dim_best_split]
        self.dupl_caused += self.delta_dupl_best_split

        return self.best_split, self.top_score, self.dim_best_split, self.delta_dupl_best_split

    def apply_best_split(self, band_condition):  # we only copy T
        if self.regular_partition:
            self.sample_S.sort(key=lambda x: x[self.dim_best_split])
            self.sample_T.sort(key=lambda x: x[self.dim_best_split])

            p_new_1_sample_S = []
            p_new_1_sample_T = []
            p_new_2_sample_S = []
            p_new_2_sample_T = []
            for i in range(len(self.sample_S)):
                if self.sample_S[i][self.dim_best_split] < self.best_split:
                    p_new_1_sample_S.append(self.sample_S[i])
                else:
                    p_new_2_sample_S.append(self.sample_S[i])
            # now ad all duplicates but only duplicatre relation T
            biggest_value_of_S_tuple_in_p_new_1 = self.A[self.dim_best_split][1]
            smallest_value_of_S_tuple_in_p_new_2 = self.A[self.dim_best_split][0]
            if len(p_new_1_sample_S) >= 1:
                biggest_value_of_S_tuple_in_p_new_1 = p_new_1_sample_S[-1][self.dim_best_split]
            if len(p_new_2_sample_S) >= 1:
                smallest_value_of_S_tuple_in_p_new_2 = p_new_2_sample_S[0][self.dim_best_split]

            # now add all the tuple Int + tuple in band that belong to T

            for tuple in self.sample_T:
                if tuple[self.dim_best_split] <= biggest_value_of_S_tuple_in_p_new_1 + band_condition[self.dim_best_split]:
                    p_new_1_sample_T.append(tuple)
                if tuple[self.dim_best_split] >= smallest_value_of_S_tuple_in_p_new_2 - band_condition[self.dim_best_split]:
                    p_new_2_sample_T.append(tuple)


            p_new_1_A = self.A.copy()
            p_new_1_A[self.dim_best_split] = (self.A[self.dim_best_split][0], self.best_split)
            p_new_2_A = self.A.copy()
            p_new_2_A[self.dim_best_split] = (self.best_split, self.A[self.dim_best_split][1])




            p_new_1_sample_output = compute_output(p_new_1_sample_S, p_new_1_sample_T, band_condition)

            p_new_2_sample_output = compute_output(p_new_2_sample_S, p_new_2_sample_T, band_condition)


            p_new_1 = Partition(p_new_1_A, p_new_1_sample_S, p_new_1_sample_T, p_new_1_sample_output)
            p_new_2 = Partition(p_new_2_A, p_new_2_sample_S, p_new_2_sample_T, p_new_2_sample_output)
            return p_new_1, p_new_2
        else:
            self.sub_partitions[self.dim_best_split] += 1
            self.nr_of_regions = math.prod(self.sub_partitions)

    def duplication_caused_by_small_partitioning(self):      #solution for n dimensiosn, not only rows and cols like in paper
        delta_dupl_all_dim = []
        for i in range(len(self.sub_partitions)):
            #duplication caused by having n subpartitions assuming uniform distribution in a small-bucket
            n = self.sub_partitions[i]
            dupl_caused_now = (n-1)*1/2
            tmp = n - (n % 2)
            number_of_duplicated_small_zones = tmp/2 * (tmp/2+1) - (1 - n % 2) * tmp/2
            dupl_caused_now += number_of_duplicated_small_zones/n
            dupl_caused_after_increase = n*1/2
            tmp = (n+1) - ((n+1) % 2)
            number_of_duplicated_small_zones = tmp/2 * (tmp/2+1) - (1 - (n+1) % 2) * tmp/2
            dupl_caused_after_increase += number_of_duplicated_small_zones/(n+1)
            delta_dupl = (dupl_caused_after_increase - dupl_caused_now)*len(self.sample_T)
            delta_dupl_all_dim.append(delta_dupl)
        return delta_dupl_all_dim

    def delta_Var_caused_by_small_partitioning(self):
        delta_var_all_dim = []
        nr_of_regions_now = self.nr_of_regions
        load_induced_by_1_region_now = self.get_load()[0]
        variance_now = nr_of_regions_now * load_induced_by_1_region_now**2            #when tuples are distributed uniformly sumation over regions becomes multiplication with nr of regions
        for i in range(len(self.sub_partitions)):
            nr_of_regions_after_increase = nr_of_regions_now * (self.sub_partitions[i]+1)/self.sub_partitions[i]
            load_induced_by_1_region_after = self.get_load()[0]*nr_of_regions_now/nr_of_regions_after_increase
            variance_after_increase = nr_of_regions_after_increase * load_induced_by_1_region_after**2
            delta_var = variance_now - variance_after_increase
            delta_var_all_dim.append(delta_var)
        return delta_var_all_dim


    def get_valid_dims(self, band_condition):
        valid_dims = []
        for i in range(len(self.A)):
            if self.A[i][1] - self.A[i][0] > band_condition[i] * 2:
                valid_dims.append(i)
        return valid_dims

    def get_load(self):
        load_whole_partition = (4 * (self.get_input_size()+self.dupl_caused) + 1 * self.get_output_size())     # beta 2 = 4, beta 3 = 1, relaztion sample:real = 1:1000
        if self.regular_partition:
            return [load_whole_partition, ]
        else:
            return [load_whole_partition/self.nr_of_regions, ]*self.nr_of_regions

    def get_input_size(self):
        return len(self.sample_input)

    def get_output_size(self):
        return len(self.sample_output)

    def get_topScore(self):
        return self.top_score

    def get_best_split(self):
        return self.best_split

    def get_dupl_caused_by_split(self):
        return self.delta_dupl_best_split

    def get_A(self):
        return self.A


def compute_output(S, T, band_conditions):
    output = []
    for s_element in S:
        for t_element in T:
            joins = True
            for i in range(len(band_conditions)):
                if not (abs(s_element[i] - t_element[i]) <= band_conditions[i]):
                    joins = False
            if joins:
                output.append((s_element, t_element))
    return output


def draw_random_sample(R, k, S):  # Generates k random tuples, gets replaced by random sample of table function later
    sample = []
    for i in range(k):  # (age, loc_x, loc_y, name, 0 for S, 1 for T
        sample.append((random.randint(0, 1000), random.randint(0, 1000), random.randint(0, 1000), i, S))
    return sample


def find_top_score_partition(partitions):  # should get changed to priority queue but is fast enough to n ot matter
    top_score_partition = None
    score = 0
    for p in partitions:
        if p.get_topScore() > score:
            score = p.get_topScore()
            top_score_partition = p

    if top_score_partition is None:
        print("NO SCORE IS OVER 0")
    return top_score_partition


def compute_max_worker_load(partitions, w):
    worker_loads = [0, ] * w
    partition_loads = []
    for p in partitions:
        load_of_p = p.get_load()
        for sub_partition_load in load_of_p:
            partition_loads.append(sub_partition_load)
    partition_loads.sort(reverse=True)
    for load in partition_loads:
        worker_loads[worker_loads.index(min(worker_loads))] += load

    return max(worker_loads)


def construct_pareto_data(size, S):
    a, m = 1.5, 15.  # shape and mode
    x = (np.random.pareto(a, size)) * m
    y = (np.random.pareto(a, size)) * m
    data = []
    for i in range(len(x)):
        x_tmp = min(100000, x[i])
        y_tmp = min(100000, y[i])

        data.append((x_tmp, y_tmp, 2, 1, S))
    return data


def recPart(S, T, band_condition, k, w):  # condition = epsilon for each band-join-dimension e.g. (10, 100, 100) for
    # 10 years apart, 100km ind x and y direction

    # choose distribution, pareto doesnt work well yet, cause no 1 bucket
    random_sample_S = construct_pareto_data(k // 2, 0)  # draw_random_sample(S, k//2, 0)
    random_sample_T = construct_pareto_data(k // 2, 1)  # draw_random_sample(T, k//2, 1)
    # random_sample_S = draw_random_sample(S, k//2, 0)        #
    # random_sample_T = draw_random_sample(T, k//2, 1)

    random_output_sample = compute_output(random_sample_S, random_sample_T, band_condition)
    print("random output sample:" + str(len(random_output_sample)))

    partitions = []         # all partitions
    A = []          #compute initial domain dynamically
    for dim in range(len(band_condition)):  #len(band_conditions) == number of dimensions
        A.append((0, max(random_sample_S+random_sample_T, key=lambda item:item[dim])[dim]))
    # A = [(0, 1000), (0, 1000)]  # domain 2 dimensions


    root_p = Partition(A, random_sample_S, random_sample_T, random_output_sample)
    partitions.append(root_p)
    root_p.find_best_split(partitions, band_condition, w)

    draw_samples(random_sample_S, random_sample_T)

    l_zero = load(k, len(random_output_sample), 4, 1) / w  # lower bound for worker load
    l_max = compute_max_worker_load(partitions, w)
    overhead_worker_load = (l_max - l_zero) / l_zero
    total_input = k  # since there is 0 duplication yet -> total input is k tuples (k = lowerbound of input)
    overhead_input_dupl = (total_input - k) / k

    all_partitions = []
    all_partitions.append(partitions.copy())
    over_head_history = []
    over_head_history.append(max(overhead_input_dupl, overhead_worker_load))
    full_overhead_history = []
    full_overhead_history.append((overhead_input_dupl, overhead_worker_load))

    termination_condition = True
    i = 0
    while termination_condition:
        p_max = find_top_score_partition(partitions)
        if p_max is None:
            print("p_max ios None")
            break
        total_input += p_max.get_dupl_caused_by_split()
        if p_max.regular_partition:
            partitions.remove(p_max)
            p_new_1, p_new_2 = p_max.apply_best_split(band_condition)
            p_new_1.find_best_split(partitions, band_condition, w)
            p_new_2.find_best_split(partitions, band_condition, w)

            partitions.append(p_new_1)
            partitions.append(p_new_2)
        else:   #1 bucket --> partition stays the same, just adds row/col

            p_max.apply_best_split(band_condition)
            p_max.find_best_split(partitions, band_condition, w)


        l_max = compute_max_worker_load(partitions, w)
        overhead_worker_load = (l_max - l_zero) / l_zero
        overhead_input_dupl = (total_input - k) / k


        full_overhead_history.append((overhead_input_dupl, overhead_worker_load))       #jsut to collect data
        over_head_history.append(max(overhead_input_dupl, overhead_worker_load))

        if overhead_input_dupl > overhead_worker_load:
            termination_condition = False
            print(i)
        all_partitions.append(partitions.copy())
        i += 1
        if i == 200:
            break
    print("iteration")
    print(i)

    best_partition_overhead = min(over_head_history)
    index_of_best_partition = over_head_history.index(best_partition_overhead)

    return random_sample_S, random_sample_T, all_partitions[0:index_of_best_partition], total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, full_overhead_history


def draw_partitions(S, T, parts):
    p = figure(plot_width=1000, plot_height=600)
    count = 1
    colors = bp.viridis(len(parts))
    for el in parts:
        start_x = []
        end_x = []
        start_y = []
        end_y = []

        for part in el:
            partition = part.get_A()
            start_x.append(partition[0][0])
            end_x.append(partition[0][1])
            start_y.append(partition[1][0])
            end_y.append(partition[1][1])

        width = [x1 - x2 for x1, x2 in zip(end_x, start_x)]
        height = [y1 - y2 for y1, y2 in zip(end_y, start_y)]
        center_x = [(x1 + x2) / 2 for x1, x2 in zip(end_x, start_x)]
        center_y = [(y1 + y2) / 2 for y1, y2 in zip(end_y, start_y)]

        part_names = []
        for i in range(len(center_x)):
            part_names.append("P{}".format(count))

        p.rect(x=center_x[-1], y=center_y[-1], width=width[-1],
               height=height[i], fill_color=colors[i], line_color=colors[i], legend_label=part_names[i],
               name=part_names[i], visible=False)

        count += 1
        p.legend.click_policy = "hide"
        hover = HoverTool(tooltips=[("name", "$name"), ("x", "$x"), ("y", "$y")])

    p.add_tools(hover)

    for i in range(len(S)):
        p.cross(x=S[i][0], y=S[i][1], line_color="blue")

    for i in range(len(T)):
        p.cross(x=T[i][0], y=T[i][1], line_color="black")

    show(p)


def draw_samples(S, T):
    p = figure(plot_width=1000, plot_height=600)
    count = 1
    for i in range(len(S)):
        p.cross(x=S[i][0], y=S[i][1], line_color="blue")

    for i in range(len(T)):
        p.cross(x=T[i][0], y=T[i][1], line_color="black")

    show(p)

if __name__ == '__main__':
    s, t, parts, total_input, l_max, overhead_input_dupl, overhead_worker_load, l_zero, over_head_history = recPart(1, 2, [50, 50], 1000, 10)
    print(parts)
    print("-----")
    print("Min input: 1000")
    print("total input:" + str(total_input))
    print("overhead input" + str(overhead_input_dupl))
    print("---load")
    print("min wordload per machine: " + str(l_zero))
    print("woardload of worst machine: " + str(l_max))
    print("workload overhead: " + str(overhead_worker_load))
    print(over_head_history)
    draw_partitions(s, t, parts)

