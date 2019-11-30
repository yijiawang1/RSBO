import REP_miso_gw as miso_gw
import REP_miso_ky as miso_ky
# import REP_miso_pd as miso_pd
import REP_miso_mc as miso_mc
import REP_miso_gwItem as miso_gwItem


def identify_problem(argv, bucket):
    benchmark_name = argv[0]
    prob_name = argv[-1]
    decode = benchmark_name.split("_")
    print('argv = ', argv)
    print(decode)
    print('benchmark_name = ', benchmark_name)
    print('prob_name = ', prob_name)
    if decode[0] == "miso":
        which_problem = int(argv[1])
        replication_no = int(argv[2])
        print('which_problem = ', which_problem)
        print('replication_no = ', replication_no)

        if decode[1] == "gw": problem_class = miso_gw.class_collection[benchmark_name]

        elif decode[1] == "ky": problem_class = miso_ky.class_collection[benchmark_name]

        # elif decode[1] == "pd": problem_class = miso_pd.class_collection[benchmark_name]

        elif decode[1] == "mc": problem_class = miso_mc.class_collection[benchmark_name]

        elif decode[1] == "it": problem_class = miso_gwItem.class_collection[benchmark_name]

        else: raise ValueError("func name not recognized")

        problem = problem_class(replication_no, which_problem, bucket, prob_name)

    else:
        raise ValueError("task name not recognized")
    return problem