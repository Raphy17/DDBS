import random


def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(100):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample

a = draw_random_sample(0, 100, 0)
b = draw_random_sample(0, 100, 1)


def find_dupl(a, i, eps, dim):
    dupl = 0
    v_i_plus_1 = a[i+1][dim]
    v_i = a[i][dim]
    j = i
    while j >=0:
        if v_i_plus_1 - a[j][dim] > eps:
            break
        j -= 1
        dupl += 1
    j = i+1
    while j <= len(a) -1:
        if a[j][dim] - v_i > eps:
            break
        j += 1
        dupl += 1
    return dupl


def best_split(partitions, w, a, dimensions):
    bestSplit = None
    topScore = 0
    Vp = 50  # before applying partitioning
    for dim in range(dimensions):  # find best split out of all dimensions
        best_x = 0
        score_best_x = 0
        a.sort(key=lambda x: x[dim])  # sort input_sample on dimension A

        for i in range(0, len(a) - 2):  # find best split a single dimension
            x = (a[i][dim] + a[i + 1][dim]) / 2
            delta_var_x = 1000 #ask on monday
            delta_dup_x = find_dupl(a, i, 5, dim)

            if delta_dup_x == 0:
                delta_dup_x = 1
            score_x = delta_var_x / delta_dup_x

            if score_x > score_best_x:
                score_best_x = score_x
                best_x = x
        if score_best_x > topScore:
            topScore = score_best_x
            bestSplit = best_x
    return bestSplit, topScore

test = [(9, 3), (2, 5), (1, 6), (4, 7)]
print(best_split(1, 1, a, 3))