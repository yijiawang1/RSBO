__author__ = 'jialeiwang'
__author__ = 'matthiaspoloczek'

import numpy
import pandas
import scipy.stats

def compute_b(s_x, zero_x_prime_arr, noise_var, gp):
    """Compute vector a and b in h(a,b)
    :param s_x: (s, x)
    :param zero_x_prime_arr: (0, x') for all x' in all_x
    :param mu_zero_x_prime_arr: posterior mean of (0, x') for all x' in all_x
    :param var_s_x: posterior variance of (s ,x)
    :param noise_var: noise variance of (s, x)
    :param cov: covariance object
    :return: (a,b)
    """
    total_pts = numpy.vstack((s_x.reshape((1,-1)), zero_x_prime_arr))
    b = numpy.zeros(len(zero_x_prime_arr))
    for i in range(len(zero_x_prime_arr)):
        b[i] = gp.compute_variance_of_points(total_pts[[0, i+1], :])[0, 1]
    b /= numpy.sqrt(noise_var + gp.compute_variance_of_points(s_x.reshape((1,-1)))[0, 0])
    return b

def compute_c_A(a_in, b_in):
    """Algorithm 1 in Frazier 2009 paper
    :param a_in:
    :param b_in:
    :return:
    """
    M = len(a_in)
    # Use the same subscripts as in Algorithm 1, therefore, a[0] and b[0] are dummy values with no meaning
    a = numpy.concatenate(([numpy.inf], a_in))
    b = numpy.concatenate(([numpy.inf], b_in))
    c = numpy.zeros(M+1)
    c[0] = -numpy.inf
    c[1] = numpy.inf
    A = [1]
    for i in range(1, M):
        c[i+1] = numpy.inf
        while True:
            j = A[-1]
            c[j] = (a[j] - a[i+1]) / (b[i+1] - b[j])
            if len(A) != 1 and c[j] <= c[A[-2]]:
                del A[-1]
            else:
                break
        A.append(i+1)
    return c, A

def compute_kg(a, b, cost, cutoff=10.0):
    """ Algorithm 2 in Frazier 2009 paper """
    df = pandas.DataFrame({'a':a, 'b':b})
    sorted_df = df.sort_values(by=['b', 'a'])
    sorted_df['drop_idx'] = numpy.zeros(len(sorted_df))
    sorted_index = sorted_df.index
    # sorted_df.index = xrange(len(sorted_df))
    for i in xrange(len(sorted_index)-1):
        if sorted_df.ix[sorted_index[i], 'b'] == sorted_df.ix[sorted_index[i+1], 'b']:
            sorted_df.ix[sorted_index[i], 'drop_idx'] = 1
    truncated_df = sorted_df.ix[sorted_df['drop_idx']==0, ['a', 'b']]
    new_a = truncated_df['a'].values
    new_b = truncated_df['b'].values
    index_keep = truncated_df.index.values
    c, A = compute_c_A(new_a, new_b)
    if len(A) <= 1:
        return 0.0, numpy.array([])
    final_b = numpy.array([new_b[idx-1] for idx in A])
    final_index_keep = numpy.array([index_keep[idx-1] for idx in A])
    final_c = numpy.array([c[idx] for idx in A])
    # compute log h() using numerically stable method
    d = numpy.log(final_b[1:] - final_b[:-1]) - 0.5 * numpy.log(2. * numpy.pi) - 0.5 * numpy.power(final_c[:-1], 2.0)
    abs_final_c = numpy.absolute(final_c[:-1])
    for i in xrange(len(d)):
        if abs_final_c[i] > cutoff:
            d[i] += numpy.log1p(-final_c[i] * final_c[i] / (final_c[i] * final_c[i] + 1))
        else:
            d[i] += numpy.log1p(-abs_final_c[i] * scipy.stats.norm.cdf(-abs_final_c[i]) / scipy.stats.norm.pdf(abs_final_c[i]))
    return numpy.exp(d).sum() / cost, final_index_keep, abs_final_c

def compute_grad_b(s_x, zero_x_prime_arr, noise_var, gp, index_keep):
    grad_b = numpy.zeros((len(index_keep), len(s_x)))    # shape number of useful b's \times dim of s_x
    reshaped_s_x = s_x.reshape((1,-1))
    total_pts = numpy.vstack((reshaped_s_x, zero_x_prime_arr[index_keep, :]))
    total_var_s_x = noise_var + gp.compute_variance_of_points(reshaped_s_x)[0,0]
    for i in range(len(index_keep)):
        grad_b[i, :] = total_var_s_x * gp.compute_grad_variance_of_points(total_pts[[0, i+1], :], num_derivatives=1)[0,0,1,:] - \
                       0.5 * gp.compute_variance_of_points(total_pts[[0, i+1], :])[0,1] * gp.compute_grad_variance_of_points(reshaped_s_x, num_derivatives=1)[0,0,0,:]
    return grad_b[:, 1:] / numpy.power(total_var_s_x, 1.5)

def compute_kg_and_grad(s, x, search_domain, num_discretization, noise_var, cost, gp):
    all_x_prime = search_domain.generate_uniform_random_points_in_domain(num_discretization)
    all_zero_x_prime = numpy.hstack((numpy.zeros((num_discretization,1)), all_x_prime))
    a = gp.compute_mean_of_points(all_zero_x_prime)
    s_x = numpy.concatenate(([s], x))
    b = compute_b(s_x, all_zero_x_prime, noise_var, gp)
    kg, index_keep, abs_c = compute_kg(a, b, cost)
    grad_b = compute_grad_b(s_x, all_zero_x_prime, noise_var, gp, index_keep)
    gradient = numpy.zeros(len(x))
    for i in range(len(abs_c)):
        gradient += (grad_b[i,:] - grad_b[i+1,:]) * scipy.stats.norm.pdf(abs_c[i])
    return kg, gradient / cost

def compute_kg_and_grad_given_x_prime(s, x, all_zero_x_prime, noise_var, cost, gp):
    a = gp.compute_mean_of_points(all_zero_x_prime)
    s_x = numpy.concatenate(([s], x))
    b = compute_b(s_x, all_zero_x_prime, noise_var, gp)

    if numpy.all(numpy.abs(b) <= 1e-6):
        return 0.0, numpy.zeros(len(x)) # as kg, gradient / cost

    kg, index_keep, abs_c = compute_kg(a, b, cost)
    grad_b = compute_grad_b(s_x, all_zero_x_prime, noise_var, gp, index_keep)
    gradient = numpy.zeros(len(x))
    for i in range(len(abs_c)):
        gradient += (grad_b[i,:] - grad_b[i+1,:]) * scipy.stats.norm.pdf(abs_c[i])
    return kg, gradient / cost

def compute_kg_given_x_prime(s, x, all_zero_x_prime, noise_var, cost, gp):
    '''
    Return KG/cost for point x at s
    :param s: the max_step
    :param x: the point for which to compute KG/cost
    :param all_zero_x_prime: the possible locations of the posterior optimum
    :param noise_var: the observational noise of the query
    :param cost: the cost of querying s at point x
    :param gp: The Gp object
    :return: KG/cost for point x at s
    '''
    a = gp.compute_mean_of_points(all_zero_x_prime)
    s_x = numpy.concatenate(([s], x))
    b = compute_b(s_x, all_zero_x_prime, noise_var, gp)

    # compute_kg(a, b, cost) would throw a valueerror if b is zero everywhere
    # (for zero covariance, e.g., due to bad hyperparameters or distant points)
    if numpy.all(numpy.abs(b) <= 1e-6):
        print 'Warning: compute_kg_given_x_prime: all bs are zero for s ' + str(s)
        # Often caused by bad hyperparameters. If it occurs for all s and many sets of initial points, increase
        # the number of initial points or enforce stricter bounds on length scales via the hyperprior.
        kg = 0.0
    else:
        kg, index_keep, abs_c = compute_kg(a, b, cost)
    return kg
