import numpy as np

def construct_pareto_data(size, z, S, dim):
    values = []
    a, m = z, 1  # shape and mode
    for i in range(dim):
        x = (np.random.pareto(a, size)+1) * m
        values.append(x)

    data = []
    for i in range(len(x)):
        t = []
        for d in range(dim):
            t.append(values[d][i])
        t.append(i)
        t.append(S)
        data.append(tuple(t))
    return data


def construct_normal_data(size, S):
    mu, sigma = 50, 15
    x = np.random.normal(mu, sigma, size)
    y = np.random.normal(mu, sigma, size)
    z = np.random.normal(mu, sigma, size)
    data = []
    for i in range(len(x)):
        x_tmp = min(100, x[i])
        y_tmp = min(100, y[i])
        z_tmp = min(100, z[i])
        data.append((x_tmp, y_tmp, z_tmp, i, S))
    return data


def construct_uniform_data(k, S):  # Generates k random tuples, gets replaced by random sample of table function later
    sample = []
    for i in range(k):  # (age, loc_x, loc_y, name, 0 for S, 1 for T
        sample.append((random.randint(0, 1000), random.randint(0, 1000), random.randint(0, 1000), i, S))
    return sample
