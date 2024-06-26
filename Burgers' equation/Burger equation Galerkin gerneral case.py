# coding by zhaolong deng
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, eval_hermitenorm


def determine_validindex(k, degreeM):
    assert (k <= degreeM)
    result_coefficient_matrix = np.zeros((degreeM + 1, degreeM + 1))
    for i in range(0,degreeM + 1):
        for j in range(0,degreeM + 1):
            s = (i + j + k)/2
            if ((i + j + k) % 2) == 0:
                if max(i, j, k) <= s:
                    result_coefficient_matrix[i,j] = (np.math.factorial(i) * np.math.factorial(j) * np.math.factorial(k))/\
                                                     (np.math.factorial(s - i) * np.math.factorial(s - j) * np.math.factorial(s - k) * np.math.factorial(k))
    return result_coefficient_matrix

def Runge_Kutta_method(y_last, k1, k2, k3, k4, delta_t):
    y_new = y_last + delta_t * (k1 + 2 * k2 + 2 * k3 + k4)/6
    return y_new

def d_dx(vector, delta_x):
    result_vector = np.zeros(vector.shape[0])
    result_vector[0] = (vector[1]-vector[0])/delta_x
    for i in range(1,(vector.shape[0]-1)):
        result_vector[i] = (vector[i+1]-vector[i-1])/(2*delta_x)
    result_vector[-1] = (vector[-1]-vector[-2])/delta_x

    return result_vector

def Second_d_dx(u, delta_x):
    result_vec = np.zeros(u.shape[0])
    result_vec[0] = (u[2] - 2 * u[1] + u[0])/delta_x**2
    for i in range(1, u.shape[0] - 1):
        result_vec[i] = (u[i + 1] - 2 * u[i] + u[i - 1])/delta_x**2
    result_vec[-1] = (u[-1] - 2 * u[-2] + u[-3])/delta_x**2
    return result_vec

def viscosty_term(v, uslide, delta_x):
    return v * Second_d_dx(uslide, delta_x)

def revise_sequence(sequence, n):
    revised_sequence = [eval_hermite(n, x) for x in sequence]
    return revised_sequence

def generate_derivative_matrix(imput_vec_matrix, delta_x):
    derivative_matrix = np.zeros(imput_vec_matrix.shape)
    for i in range(imput_vec_matrix.shape[1]):
        derivative_matrix[:, i] = d_dx(imput_vec_matrix[:, i], delta_x)
    return derivative_matrix

def RK_helper_Galerkin(input_vec_matrix, delta_x, delta_t, degreeM, v):
    assert( input_vec_matrix.shape[1] == degreeM + 1)
    derivative_matrix_x = generate_derivative_matrix(input_vec_matrix, delta_x)
    # Calculating the derivative on each point with a given matrix
    # Generating the coefficient matrix respect to the index
    result_matrix = np.zeros(input_vec_matrix.shape)

    for i in range(input_vec_matrix.shape[0]):
        # Iterating columns over the input_matrix
        u_vec_slice = input_vec_matrix[i, :]
        derivative_slice = derivative_matrix_x[i, :]
        for j in range(degreeM + 1):
            # Iterating over all u with Runge-Kutta
            coefficient_matrix = determine_validindex(j, degreeM)
            u_last = u_vec_slice[j]
            u_k = u_vec_slice.copy()

            viscosity = viscosty_term(v, input_vec_matrix[:, j], delta_x)[i]
            # Copy for better iteration, which do not affect in latter process
            k1 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
            u_k[j] += 0.5 * delta_t
            k2 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
            u_k[j] += + 0.5 * delta_t
            k3 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
            u_k[j] += delta_t
            k4 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
            result = Runge_Kutta_method(u_last, k1, k2, k3, k4, delta_t)
            result_matrix[i, j] = result
    return result_matrix


def Galerkin_computing(x_start, x_end, t_start, t_end, num_x, num_t, degreeM, v):
    original_matrix = np.zeros((num_x, num_t, degreeM + 1))
    # Containing degreeM of u-matrix, which is a 3_D space
    final_matrix = np.zeros((num_x, num_t))
    x_vals = np.linspace(x_start, x_end, num_x)
    t_vals = np.linspace(t_start, t_end, num_t)
    delta_x = (x_start - x_end) / num_x
    delta_t = (t_start - t_end) / num_t
    original_matrix[:, 0, 1] = np.sin(x_vals)
    random_num = [0.05]
    for i in range(0, num_t - 1):
        # Iterating for u_1 though u_M
        input_matrix = original_matrix[: , i, :]
        result_matrix = RK_helper_Galerkin(input_matrix, delta_x, delta_t, degreeM, v)
        original_matrix[:, i + 1,:] = result_matrix
    for j in range(degreeM + 1):
        # Computing for the final matrix
        final_matrix += original_matrix[:, :, j] * revise_sequence(random_num, j)[0]
    final_matrix = np.transpose(final_matrix)
    X, T = np.meshgrid(x_vals, t_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, final_matrix, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    # ax.set_zlim3d(-0.2, 0.2)
    plt.show()
    return final_matrix

Galerkin_computing(0,3,0,0.2, 200, 60, 5, 0)