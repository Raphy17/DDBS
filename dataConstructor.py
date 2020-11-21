import numpy as np

def construct_pareto_data(size, z, S, dim):
    values = []
    a, m = z, 1  # shape and mode
    for i in range(dim):
        x = (np.random.pareto(a, size)+1) * m
        values.append(x)

    data = []
    for i in range(size):
        t = []
        for d in range(dim):
            t.append(values[d][i])
        t.append(i)
        t.append(S)
        data.append(tuple(t))
    return data


def construct_normal_data(size, mu, sigma, S, dim):
    values = []
    for i in range(dim):
        x = np.random.normal(mu, sigma, size)
        values.append(x)

    data = []
    for i in range(size):
        t = []
        for d in range(dim):
            t.append(values[d][i])
        t.append(i)
        t.append(S)
        data.append(tuple(t))
    return data


def construct_uniform_data(k, S):  # Generates k random tuples, gets replaced by random sample of table function later
    sample = []
    for i in range(k):  # (age, loc_x, loc_y, name, 0 for S, 1 for T
        sample.append((random.randint(0, 1000), random.randint(0, 1000), random.randint(0, 1000), i, S))
    return sample
