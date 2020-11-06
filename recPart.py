import random
import pandas as pd
from math import pi
import numpy as np
from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Line, HoverTool, FactorRange
import bokeh.palettes as bp


def load(input_size, output_size, b2, b3):
    return b2 * input_size + b3 * output_size


def per_worker_load_variance(partitions, w):  # uses for beta 2, beta 3: (4, 1) (like in amazon cloud cluster)
    Vp = (w - 1) / w ** 2
    tmp = 0
    for p in partitions:
        load_of_p = p.get_load()
        tmp += load_of_p ** 2
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
        self.dupl_best_split = None

    def __repr__(self):

        return "{}".format(self.A)

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
                    x = (self.sample_input[i][dim] + self.sample_input[i + 1][dim]) / 2  #
                    delta_dup_x = find_dupl(self.sample_input, i, band_condition[dim], dim)  # replace 5 with condition
                    Vp_new = Vp - (w - 1) / w ** 2 * (self.get_load() ** 2)
                    Vp_new += (w - 1) / w ** 2 * (load(1 + i + delta_dup_x, 1 + i + delta_dup_x, 4, 1) ** 2 + load(
                        len(self.sample_input) - 1 - i + delta_dup_x, len(self.sample_input) - 1 - i + delta_dup_x, 4,
                        1) ** 2)
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
            self.dupl_best_split = dupl_best_split
        else:
            # TODO: Calcuate deltas

            # Caluculate sigmas

            sigma_r = delta_var_rp1_c / delta_dup_rp1_c
            sigma_c = delta_var_rp1_c / delta_dup_r_cp1
            if sigma_r > sigma_c:
                top_score = sigma_r
                best_split = "row"
            else:
                top_score = sigma_c
                best_split = 'column'

        return best_split, top_score, dim_best_split, dupl_best_split

    def apply_best_split(self, band_condition):  # we only copy T
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

    def get_valid_dims(self, band_condition):
        valid_dims = []
        for i in range(len(self.A)):
            if self.A[i][1] - self.A[i][0] > band_condition[i] * 2:
                valid_dims.append(i)
        return valid_dims

    def get_load(self):
        return 4 * self.get_input_size() + 1 * self.get_output_size()  # beta 2 = 4, beta 3 = 1

    def get_input_size(self):
        return len(self.sample_input)

    def get_output_size(self):
        return len(self.sample_output)

    def get_topScore(self):
        return self.top_score

    def get_best_split(self):
        return self.best_split

    def get_dupl_caused_by_split(self):
        return self.dupl_best_split

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
        sample.append((random.randint(0, 100), random.randint(0, 100), random.randint(0, 1000), i, S))
    return sample


def find_top_score_partition(partitions):  # should get changed to priority queue but is fast enough to n ot matter
    top_score_partition = None
    score = 0
    for p in partitions:
        if p.get_topScore() > score:
            score = p.get_topScore()
            top_score_partition = p
    return top_score_partition


def compute_max_worker_load(partitions, w):
    worker_loads = [0, ] * w
    partition_loads = []
    for p in partitions:
        partition_loads.append(p.get_load())
    partition_loads.sort(reverse=True)
    for load in partition_loads:
        worker_loads[worker_loads.index(min(worker_loads))] += load

    return max(worker_loads)


def construct_pareto_data(size, S):
    a, m = 0.5, 15.  # shape and mode
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
    #random_sample_S = construct_pareto_data(k // 2, 0)  # draw_random_sample(S, k//2, 0)
   # random_sample_T = construct_pareto_data(k // 2, 1)  # draw_random_sample(T, k//2, 1)
    random_sample_S = draw_random_sample(S, k//2, 0)        #
    random_sample_T = draw_random_sample(T, k//2, 1)

    random_output_sample = compute_output(random_sample_S, random_sample_T, band_condition)

    partitions = []         # all partitions
    #A = [(0, 100000), (0, 100000)]  # because our random samples have values in between these domains
    A = [(0, max(random_sample_S+random_sample_T, key=lambda item:item[0])[0]), (0,  max(random_sample_S+random_sample_T, key=lambda item:item[1])[1])]
    print(A)

    root_p = Partition(A, random_sample_S, random_sample_T, random_output_sample)
    partitions.append(root_p)
    print(root_p.find_best_split(partitions, band_condition, w))

    draw_samples(random_sample_S, random_sample_T)

    l_zero = load(k, len(random_output_sample), 4, 1) / w  # lower bound for worker load
    l_max = compute_max_worker_load(partitions, w)
    overhead_worker_load = (l_max - l_zero) / l_zero
    total_input = k  # since there is 0 duplication yet -> total input is k tuples (k = lowerbound of input)
    overhead_input_dupl = (total_input - k) / k

    all_partitions = []
    all_partitions.append(partitions.copy())
    print(root_p)
    termination_condition = True
    i = 0
    while termination_condition:
        p_max = find_top_score_partition(partitions)
        if p_max is None:
            break
        total_input += p_max.get_dupl_caused_by_split()
        partitions.remove(p_max)
        p_new_1, p_new_2 = p_max.apply_best_split(band_condition)
        p_new_1
        p_new_2
        p_new_1.find_best_split(partitions, band_condition, w)
        p_new_2.find_best_split(partitions, band_condition, w)
        print(p_new_1.get_topScore())
        print(p_new_2.get_topScore())
        partitions.append(p_new_1)
        partitions.append(p_new_2)
        l_max = compute_max_worker_load(partitions, w)
        overhead_worker_load = (l_max - l_zero) / l_zero
        overhead_input_dupl = (total_input - k) / k
        if overhead_input_dupl > overhead_worker_load:
            termination_condition = False

        all_partitions.append(partitions.copy())
        i += 1
        if i == 50:
            break

    return random_sample_S, random_sample_T, all_partitions


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
    s, t, parts = recPart(1, 2, [5, 5], 500, 10)
    print(parts[-1])
    draw_partitions(s, t, parts)

