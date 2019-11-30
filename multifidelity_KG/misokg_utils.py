import sys
from moe.optimal_learning.python.data_containers import HistoricalData
import numpy as np
import pickle
from pickle import dump
from joblib import Parallel, delayed

np.set_printoptions(threshold=sys.maxint)

# ================================================================================================= #
#                                     generate initial data                                         #
#                                         (multiple IS)                                             #
# ================================================================================================= #
def sample_initial_data(problem, num_initial_pts_per_IS, exp_path, result_folder):
    points = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_initial_pts_per_IS)
    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim() + 1)  # increased by one for index of IS
    for IS in problem.obj_func_min.getList_IS_to_query():
        points_dict[IS] = np.hstack((IS * np.ones(num_initial_pts_per_IS).reshape((-1, 1)), points))
        random_seed = np.random.randint(1000)
        vals_dict[IS] = np.array([-1.0*problem.obj_func_min.evaluate(IS,pt,random_seed,exp_path) for pt in points])
        noise_dict[IS] = np.ones(len(points)) * problem.obj_func_min.noise_and_cost_func(IS, None)[0]
        # note: misoKG will learn the noise from sampled data
        new_historical_data.append_historical_data(points_dict[IS], vals_dict[IS], noise_dict[IS])
    return new_historical_data


def sample_initial_data_grid(problem, pts_per_dimension, num_initial_pts_per_IS, exp_path, result_folder):
    points = problem.obj_func_min.get_moe_domain().generate_grid_points_in_domain(pts_per_dimension)
    id_points = np.random.randint(pts_per_dimension**problem.obj_func_min.getDim(), size=num_initial_pts_per_IS)
    points = points[id_points,:]
    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim() + 1)  # increased by one for index of IS
    for IS in problem.obj_func_min.getList_IS_to_query():
        points_dict[IS] = np.hstack((IS * np.ones(num_initial_pts_per_IS).reshape((-1, 1)), points))
        random_seed = np.random.randint(1000)
        vals_dict[IS] = np.array([-1.0*problem.obj_func_min.evaluate(IS,pt,random_seed,exp_path) for pt in points])
        noise_dict[IS] = np.ones(len(points)) * problem.obj_func_min.noise_and_cost_func(IS, None)[0]
        # note: misoKG will learn the noise from sampled data
        new_historical_data.append_historical_data(points_dict[IS], vals_dict[IS], noise_dict[IS])
    return new_historical_data


def sample_initial_data_test_IS(problem, num_initial_pts_per_IS, exp_path, result_path):
    list_init_pts_value = []
    points_flag1 = problem.obj_func_min.get_moe_domain().generate_uniform_flag1_IS_domain(num_initial_pts_per_IS)
    points_flag2 = problem.obj_func_min.get_moe_domain().generate_uniform_flag2_IS_domain(num_initial_pts_per_IS)
    # points_width = np.array([[5,10]])

    order_flag1, order_flag2 = np.meshgrid(np.arange(num_initial_pts_per_IS), np.arange(num_initial_pts_per_IS))
    order_flag1 = order_flag1.reshape(-1)
    order_flag2 = order_flag2.reshape(-1)

    points_flag1_v = points_flag1[order_flag1,:]
    points_flag2_v = points_flag2[order_flag2,:]
    points = np.hstack((points_flag1_v.reshape((-1,2)), points_flag2_v.reshape((-1,2))))
    # points_width_v = np.repeat(points_width, np.shape(points_flag)[0], axis=0)
    # points = np.hstack((points_flag, points_width_v))

    points_dict = {}
    vals_dict = {}
    noise_dict = {}
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim() + 1)  # increased by one for index of IS
    for IS in problem.obj_func_min.getList_IS_to_query():
        points_dict[IS] = np.hstack((IS * np.ones((np.shape(points)[0], 1)), points))
        random_seed = np.random.randint(1000)
        vals_dict[IS] = np.array([-1.0 * problem.obj_func_min.evaluate(IS, pt, random_seed, exp_path) for pt in points])
        noise_dict[IS] = np.ones(len(points)) * problem.obj_func_min.noise_and_cost_func(IS, None)[0]
        new_historical_data.append_historical_data(points_dict[IS], vals_dict[IS], noise_dict[IS])

        pts_value = np.hstack((points, vals_dict[IS].reshape((-1,1)) ))
        list_init_pts_value.append(pts_value)

    with open(result_path+'_initial_samples.txt', "w") as file: 
        file.write(str(list_init_pts_value))
    with open(result_path+'_initial_samples.pickle', "wb") as file: 
        dump(list_init_pts_value, file)

    return new_historical_data, points, points_flag1, points_flag2



# ================================================================================================= #
#                                      generate initial data                                        #
#                                           (single IS)                                             #
# ================================================================================================= #

# --------------------------------------------------------------------------- #
#                              load from file                                 #
# --------------------------------------------------------------------------- #
def load_sample_data(problem, num_per_var, exp_path, result_path):
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    with open(result_path+'_initial_samples.pickle', 'rb') as file: 
        list_init_pts_value = pickle.load(file)
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
    count = -1
    # for max_step in problem.obj_func_min.getSearchDomain()[0,:]:
    s_min = problem.obj_func_min.getSearchDomain()[0, 0]
    s_max = problem.obj_func_min.getSearchDomain()[0, 1]
    for s in np.linspace(s_min, s_max, num=problem.obj_func_min.getNums()):
        count += 1
        pts_value = list_init_pts_value[count]
        points = pts_value[:, 0:-1]
        vals_array = pts_value[:, -1]
        noise_array = np.array([problem.obj_func_min.noise_and_cost_func(pt)[0] for pt in points])
        new_historical_data.append_historical_data(points, vals_array, noise_array)

    # with open(result_path+'.pickle', 'rb') as file: f_dict = pickle.load(file)
    # sampled_points = f_dict.get('sampled_points')
    # sampled_vals = f_dict.get('sampled_vals')
    # print(sampled_points[:,0])
    # print(-1.0 * sampled_vals)
    # noise_array = np.array([problem.obj_func_min.noise_and_cost_func(pt)[0] for pt in sampled_points])
    # new_historical_data.append_historical_data(sampled_points, -1.0*sampled_vals, noise_array)
    
    return new_historical_data


# --------------------------------------------------------------------------- #
#                 general function for initial data generator                 #
# --------------------------------------------------------------------------- #
def sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path):
    list_init_pts_value = []
    new_historical_data = HistoricalData(dim=problem.obj_func_min.getDim())
    s_min = problem.obj_func_min.getSearchDomain()[0, 0]
    s_max = problem.obj_func_min.getSearchDomain()[0, 1]
    for s in np.linspace(s_min, s_max, num=problem.obj_func_min.getNums()):
        random_seeds = np.random.randint(900, size=num_initial_pts_per_s)
        points = np.hstack((s * np.ones(num_initial_pts_per_s).reshape((-1, 1)), points_x))

        vals_array = np.array([-1.0*problem.obj_func_min.evaluate(pt, random_seed, exp_path) \
                              for (pt,random_seed) in zip(points, random_seeds)])

        noise_array = np.array([problem.obj_func_min.noise_and_cost_func(pt)[0] for pt in points])
        new_historical_data.append_historical_data(points, vals_array, noise_array)

        pts_value = np.hstack(( points, vals_array.reshape((-1,1)) ))
        list_init_pts_value.append(pts_value)
        with open(result_path+'_initial_samples.txt', "w") as file: 
            file.write(str(list_init_pts_value))
        with open(result_path+'_initial_samples.pickle', "wb") as file: 
            dump(np.array(list_init_pts_value), file)
    # print(list_init_pts_value)
    return new_historical_data


# --------------------------------------------------------------------------- #
#              different constraints on generating initial data               #
# --------------------------------------------------------------------------- #
def sample_initial_x_uniform(problem, num_per_var, exp_path, result_path):
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

def sample_initial_f1f2_closer_f1_further_f2(problem, num_per_var, exp_path, result_path):
    ''' flag 1 is closer to the start than flag 2 '''
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_closer_f1_further_f2(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

def sample_initial_f1f2_higher_f1_lower_f2(problem, num_per_var, exp_path, result_path):
    ''' y(f1) <= y(f2) '''
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_higher_f1_lower_f2(num_initial_pts_per_s)
    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

def sample_initial_f1f2_grid(problem, num_per_var, exp_path, result_path):
    # np.random.seed(1)
    var_dim = int(problem.obj_func_min.getDim()) - 1
    num_initial_pts_per_s = int(num_per_var * var_dim)
    points_x = problem.obj_func_min.get_moe_domain().generate_grid_f1f2_in_domain(num_initial_pts_per_s)
    # points_width = np.array([[5,10]])
    # points_width_v = np.repeat(points_width, np.shape(points_flag)[0], axis=0)
    # points_x = np.hstack((points_flag, points_width_v))

    new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
    return new_historical_data

# def sample_initial_f1f2_noconstraint_combination(problem, num_per_var, exp_path, result_path):
#     ''' combination of flag1 and flag2 '''
#     # np.random.seed(1)
#     var_dim = int(problem.obj_func_min.getDim()) - 1
#     num_initial_pts_per_s = int(num_per_var * var_dim)
#     points_flag1 = problem.obj_func_min.get_moe_domain().generate_uniform_flag1_in_domain(num_initial_pts_per_s)
#     points_flag2 = problem.obj_func_min.get_moe_domain().generate_uniform_flag2_in_domain(num_initial_pts_per_s)
#     # points_width = np.array([[5,10]])

#     order_flag1, order_flag2 = np.meshgrid(np.arange(num_initial_pts_per_s), np.arange(num_initial_pts_per_s))
#     order_flag1 = order_flag1.reshape(-1)
#     order_flag2 = order_flag2.reshape(-1)

#     points_flag1_v = points_flag1[order_flag1,:]
#     points_flag2_v = points_flag2[order_flag2,:]
#     points_x = np.hstack((points_flag1_v.reshape((-1,2)), points_flag2_v.reshape((-1,2))))
#     # points_width_v = np.repeat(points_width, np.shape(points_flag)[0], axis=0)
#     # points_x = np.hstack((points_flag, points_width_v))
    
#     new_historical_data = sample_intial_x_general(problem, num_initial_pts_per_s, points_x, exp_path, result_path)
#     return new_historical_data



# ================================================================================================= #
#                                        select start points                                        #
# ================================================================================================= #
def select_startpts_BFGS(list_sampled_points, point_to_start_from, num_multistart, problem):
    '''
    create starting points for BFGS, first select points from previously sampled points,
    but not more than half of the starting points
    :return: numpy array with starting points for BFGS
    '''
    if len(list_sampled_points) > 0:
        indices_chosen = np.random.choice(len(list_sampled_points), 
                                          int(min(len(list_sampled_points), num_multistart/2. - 1.)), 
                                          replace=False)
        start_pts = np.array(list_sampled_points)[indices_chosen]
        start_pts = np.vstack((point_to_start_from, start_pts)) # add the point that will be sampled next
    else:
        start_pts = [point_to_start_from]
    # fill up with points from an LHS
    random_pts = problem.obj_func_min.get_moe_domain().generate_uniform_random_points_in_domain(num_multistart-len(start_pts))
    start_pts = np.vstack( (start_pts, random_pts) )
    return start_pts


# --------------------------------------------------------------------------- #
#                 general function for selecting start points                 #
# --------------------------------------------------------------------------- #
def select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    '''
    create starting points for BFGS, first select points from previously sampled points,
    but not more than half of the starting points
    :return: numpy array with starting points for BFGS
    '''
    if len(list_sampled_points) > 0:
        indices_chosen = np.random.choice(len(list_sampled_points), 
                                          int(min(len(list_sampled_points), num_multistart/2.-1.)), 
                                          replace=False)
        start_pts_x = np.array(list_sampled_points)[:,1:][indices_chosen]
        start_pts_x = np.vstack((pt_x_to_start_from, start_pts_x)) # add the point that will be sampled next
    else:
        start_pts_x = [pt_x_to_start_from]
    return start_pts_x

# --------------------------------------------------------------------------- #
#               different constraints on selecting start points               #
# --------------------------------------------------------------------------- #
def select_startpts_x_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    # fill up with points from an LHS
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_uniform_x_points_in_domain(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x

def select_startpts_f1closer_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_closer_f1_further_f2(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x

def select_startpts_f1higher_BFGS(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem):
    start_pts_x = select_startpts_general(s, list_sampled_points, pt_x_to_start_from, num_multistart, problem)
    random_pts_x = problem.obj_func_min.get_moe_domain().generate_higher_f1_lower_f2(num_multistart-len(start_pts_x))
    start_pts_x = np.vstack((start_pts_x, random_pts_x))
    return start_pts_x



# ================================================================================================= #
#                                      process_parallel_results                                     #
# ================================================================================================= #
def process_parallel_results(parallel_results):
    inner_min = np.inf
    for result in parallel_results:
        if inner_min > result[1]:
            inner_min = result[1]
            inner_min_point = result[0]
    return inner_min, inner_min_point