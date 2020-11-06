from recPart import compute_output

test_1 = [
    (3, ),
    (4, ),
    (5,),
    (6,),
    (7, )
]

test_2 = [
    (10, ),
    (9, ),
    (8,),
    (7,),
    (6, )
]

print(len(compute_output(test_1, test_2, [2])))  # Should output 10
