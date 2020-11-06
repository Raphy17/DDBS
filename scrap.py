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


def construct_pareto_data(size, S):
    a, m = 2.0, 15.  # shape and mode
    x = (np.random.pareto(a, size) + 1) * m
    y = (np.random.pareto(a, size) + 1) * m
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

a = [1, 3, 4]
b = [2, 3, 4]
print(a.union(b))

