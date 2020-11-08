from recPart import compute_output, construct_pareto_data



band_conditions = [[0.00001,], [0.00002,], [0.00003,], [2, 2, 2], [4, 4, 4], [20, 20, 20, 20, 20, 20, 20, 20]]
for band_condition in band_conditions:
    output_sizes = []
    input_sizes = []
    for i in range(1, 100):
        input = i*100
        input_sizes.append(input)
        random_sample_S = construct_pareto_data(input // 2, 0, len(band_condition))
        random_sample_T = construct_pareto_data(input // 2, 1, len(band_condition))
        output = len(compute_output(random_sample_S, random_sample_T, band_condition))
        output_sizes.append(output)
    print("----")
    print(band_condition)
    print(input_sizes)
    print(output_sizes)
    print("-----")