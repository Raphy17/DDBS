import random

def draw_random_sample(R, k, S):            #Generates k random tuples, will ger replaced by random sample of table function later
    sample = []
    for i in range(k):
        sample.append((random.randint(0, 100), random.randint(0, 1000), random.randint(0, 1000), i, S))         #(age, loc_x, loc_y, name, 0 for S, 1 for T
    return sample