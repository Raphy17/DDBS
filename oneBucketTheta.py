from random import randint
from math import sqrt


def theta_join(S, T, join_condition=lambda s, t: s == t):
    """
    This implements the 1-bucket-theta algorithm found in
    http://www.ccs.neu.edu/home/mirek/papers/2011-SIGMOD-ParallelJoins.pdf

    S,T: rdds of (key, value)
    join_condition: a function that accepts two keys and returns a boolean.

    """
    ct, cs, height, width = _create_partitioning_rectangles(S, T)
    S = S.map(lambda k, v: (k, (1, v)))
    T = T.map(lambda k, v: (k, (2, v)))
    dispatch = reducer(join_condition)
    mapper = region_map(ct, cs, height, width)
    D = S.union(T).flatMap(mapper).groupByKey().flatMap(dispatch)
    return D


def _create_partitioning_rectangles(S, T):
    global cardS
    global cardT
    cardS = cardinality(S)
    cardT = cardinality(T)
    if cardT <= cardS:
        T, S = S, T
        cardT, cardS = cardS, cardT

    r = 1.0 * sc.defaultParallelism
    opt_ratio = sqrt(cardT * cardS / r)

    if (cardS % opt_ratio) == 0 and (cardT % opt_ratio) == 0:
        # optimal case
        height = width = opt_ratio
        ct, cs = int(cardT / opt_ratio), int(cardS / opt_ratio)
    elif cardS < cardT / r:
        height, width = cardS, cardT / r
        ct, cs = int(r), 1
    else:
        assert cardT / r <= cardS <= cardT
        cs = int(cardS / opt_ratio)
        ct = int(cardT / opt_ratio)
        height = width = (1 + 1. / min(cs, ct)) * opt_ratio

    return ct, cs, height, width


def row_lookup(row, ct, cs, height, width):
    start = int(row / height) + 1
    end = start + ct
    return range(start, end)


def col_lookup(col, ct, cs, height, width):
    start = int(col / width) + 1
    end = ct * cs + 1
    return range(start, end, ct)


def cardinality(X):
    return X.count()


def region_map(ct, cs, height, width):
    def mapper(x):
        (key, (source, value)) = x

        if source == 1:
            row = randint(1, cardS)
            return [(region_id, x) for region_id in row_lookup(row, ct, cs, height, width)]
        else:
            col = randint(1, cardT)
            return [(region_id, x) for region_id in col_lookup(col, ct, cs, height, width)]

    return mapper


def reducer(join_condition):
    def dispatch(x):
        # (region_id, [x1,x2,..])
        region_id, data = x

        stuples, ttuples = [], []

        for (key, (source, value)) in data:
            if source == 1:
                stuples.append((key, value))
            else:
                ttuples.append((key, value))
        return _join(ttuples, stuples, join_condition)

    return dispatch


def _join(ttuples, stuples, join_condition):
    results = []

    if len(ttuples) == 0 or len(stuples) == 0:
        return []
    for t in ttuples:
        for s in stuples:
            if join_condition(t[0], s[0]):
                keys = (t[0], s[0])
                values = (t[1], s[1])
                results.append((keys, values))
    return results