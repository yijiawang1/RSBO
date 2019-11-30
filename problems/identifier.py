import miso_gw
import miso_ky
import miso_pd
import miso_mc
import miso_gwItem


def identify_problem(argv, bucket):
    benchmark_name = argv[0]
    prob_name = argv[-1]
    decode = benchmark_name.split("_")
    print(decode)
    print('benchmark_name = ', benchmark_name)
    print('prob_name = ', prob_name)
    if decode[0] == "miso":
        which_problem = int(argv[1])
        version = int(argv[2])
        replication_no = int(argv[3])
        print('which_problem = ', which_problem)
        print('version = ', version)
        print('replication_no = ', replication_no)

        if decode[1] == "gw": problem_class = miso_gw.class_collection[benchmark_name]

        elif decode[1] == "ky": problem_class = miso_ky.class_collection[benchmark_name]

        elif decode[1] == "pd": problem_class = miso_pd.class_collection[benchmark_name]

        elif decode[1] == "mc": problem_class = miso_mc.class_collection[benchmark_name]

        elif decode[1] == "it": problem_class = miso_gwItem.class_collection[benchmark_name]

        else: raise ValueError("func name not recognized")

        problem = problem_class(replication_no, version, which_problem, bucket, prob_name)

    else:
        raise ValueError("task name not recognized")
    return problem