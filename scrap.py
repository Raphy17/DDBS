import random
import numpy as np
from bokeh.io import output_file, show, save
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Line, HoverTool, FactorRange
import bokeh.palettes as bp


def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(100):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample


def construct_pareto_data(size):
    a, m = 2.0, 15.  # shape and mode
    x = (np.random.pareto(a, size) + 1) * m
    y = (np.random.pareto(a, size) + 1) * m
    data = []
    for i in range(len(x)):
        x_tmp = min(100, x[i])
        y_tmp = min(100, y[i])

        data.append((x_tmp, y_tmp, 2, S))
    return data

def construct_normal_data(size, S):
    mu, sigma = 50, 15
    x = np.random.normal(mu, sigma, 1000)
    y = np.random.normal(mu, sigma, 1000)
    data = []
    for i in range(len(x)):
        x_tmp = min(100, x[i])
        y_tmp = min(100, y[i])
        data.append((x_tmp, y_tmp, 2, S))
    return data

def draw_samples(S, T):
    p = figure(plot_width=1000, plot_height=1000)
    count = 1
    for i in range(len(S)):
        p.cross(x=S[i][0], y=S[i][1], line_color="blue")

    for i in range(len(T)):
        p.cross(x=T[i][0], y=T[i][1], line_color="black")

    show(p)


def duplication_caused_by_small_partitioning(a):      #solution for n dimensiosn, not only rows and cols like in paper
    delta_dupl_all_dim = []
    for i in range(len(a)):
        #duplication caused by having n subpartitions assuming uniform distribution in a small-bucket
        n = a[i]

        dupl_caused_now = (n-1)*1/2
        tmp = n - (n % 2)
        number_of_duplicated_small_zones = tmp/2 * (tmp/2+1) - (1 - n % 2) * tmp/2
        dupl_caused_now += number_of_duplicated_small_zones/n
        print(dupl_caused_now)
        dupl_caused_after_increase = n*1/2
        tmp = (n+1) - ((n+1) % 2)
        number_of_duplicated_small_zones = tmp/2 * (tmp/2+1) - (1 - (n+1) % 2) * tmp/2
        dupl_caused_after_increase += number_of_duplicated_small_zones/(n+1)
        print(dupl_caused_after_increase)

        delta_dupl = (dupl_caused_after_increase - dupl_caused_now)

        delta_dupl_all_dim.append(delta_dupl)
    return delta_dupl_all_dim

def belongs_to(tuple, partitioning, dim, band_conditions): #returns the partition into which the tuple belong
    belongs_to = []

    if tuple[-1] == 0: #S tuple
        for i in range(len(partitioning)):
            is_part = True
            for d in range(dim):
                if not (partitioning[i][d][0] <= tuple[d] <= partitioning[i][d][1]):
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

a = (1, 2, 3)

print(a[1])