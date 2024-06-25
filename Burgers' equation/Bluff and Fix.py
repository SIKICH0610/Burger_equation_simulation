import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import eval_hermite
from Galerkin_Solver import GalerkinSolver, create_coeff_matrix
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def d_dx(vector, delta_x):
    """
    Calculate the first derivative of a vector using finite differences.

    Args:
        vector (numpy.ndarray): Input vector to differentiate.

    Returns:
        numpy.ndarray: Vector of first derivatives.
    """
    result_vector = np.zeros(vector.shape[0])
    result_vector[0] = (vector[1] - vector[0]) / delta_x
    for i in range(1, vector.shape[0] - 1):
        result_vector[i] = (vector[i + 1] - vector[i - 1]) / (2 * delta_x)
    result_vector[-1] = (vector[-1] - vector[-2]) / delta_x
    return result_vector

def generate_derivative_matrix(input_vec_matrix, delta_x):
    """
    Generate a matrix of derivatives for each column vector.

    Args:
        input_vec_matrix (numpy.ndarray): Matrix containing column vectors.

    Returns:
        numpy.ndarray: Matrix of first derivatives.
    """
    derivative_matrix = np.zeros(input_vec_matrix.shape)
    for i in range(input_vec_matrix.shape[1]):
        derivative_matrix[:, i] = d_dx(input_vec_matrix[:, i], delta_x)
    return derivative_matrix

#
# def bluff_and_fix_algorithm(conditions, correction_size):
#     assert len(conditions) == 8, "Conditions must include all necessary parameters."
#
#     # Compute solutions for degree M-1 (degree 4)
#     lower_degree_conditions = conditions.copy()
#     lower_degree_conditions["degreeM"] -= 1
#     lower_solver = GalerkinSolver(**lower_degree_conditions)
#     lower_result = lower_solver.solve()
#     corrected_solution_size = conditions["degreeM"] - correction_size
#     corrected_result = lower_result[:, :, :corrected_solution_size + 1]
#     result = np.copy(corrected_result)
#
#     # Initialize time step (delta_t) - define based on your time discretization
#     delta_t = (conditions['t_end'] - conditions['t_start']) / conditions['num_t']
#     delta_x = (conditions['x_end'] - conditions['x_start']) / conditions['num_x']
#     size_changed = 1
#     shape = (corrected_result.shape[0], corrected_result.shape[1], 1)
#     derivative_matrix = np.zeros(corrected_result.shape)
#     for layer in range(corrected_result.shape[2] - 1):
#         derivative_matrix[:,:, layer] = generate_derivative_matrix(corrected_result[:,:, layer], delta_x)
#
#     # Iterate through each time step and each spatial point to compute higher degree results using Runge-Kutta
#     while result.shape[2] != conditions["degreeM"] + 1:
#         coefficient_matrix = create_coeff_matrix(corrected_solution_size + size_changed,
#                                                  conditions["degreeM"])
#         higher_result = np.zeros(shape)
#         for t in range(corrected_result.shape[0] - 1):
#             for x in range(corrected_result.shape[1]):
#                 u_k = corrected_result[t, x, :]  # This is u_1 to u_4 from lower degrees
#                 u_k = np.concatenate((u_k, np.zeros(correction_size)))
#                 derivative_vector = derivative_matrix[t, x, :]
#                 derivative_vector = np.concatenate((derivative_vector, np.zeros(correction_size)))
#                 # Calculating k1 to k4 using the detailed method similar to your original RK-helper function
#                 # Runge-Kutta computations
#                 k1 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_vector)
#                 u_temp = u_k + 0.5 * delta_t * k1
#                 k2 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
#                 u_temp = u_k + 0.5 * delta_t * k2
#                 k3 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
#                 u_temp = u_k + delta_t * k3
#                 k4 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
#
#                 # Update the higher degree result
#                 higher_result[t, x, -1] = u_k[-1] + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
#
#         # new_layer_expanded = np.expand_dims(higher_result[:, :, -1], axis=2)
#         result = np.concatenate((result, higher_result), axis=2)
#         # corrected_result = result
#         # corrected_solution_size += 1
#         size_changed += 1
#
#     return result

def compute_lower_solution(conditions):
    assert len(conditions) == 8, "Conditions must include all necessary parameters."

    # Compute solutions for degree M-1
    lower_degree_conditions = conditions.copy()
    lower_degree_conditions["degreeM"] -= 1
    lower_solver = GalerkinSolver(**lower_degree_conditions)
    lower_result = lower_solver.solve()

    return lower_result


def compute_higher_solution(lower_result, conditions, correction_size):
    corrected_solution_size = conditions["degreeM"] - correction_size
    corrected_result = lower_result[:, :, :corrected_solution_size + 1]
    lower_shape = corrected_result.shape[2]

    delta_t = (conditions['t_end'] - conditions['t_start']) / conditions['num_t']
    delta_x = (conditions['x_end'] - conditions['x_start']) / conditions['num_x']
    size_changed = 0
    shape = (corrected_result.shape[0], corrected_result.shape[1], correction_size)
    corrected_result = np.concatenate((corrected_result, np.zeros(shape)), axis = 2)
    derivative_matrix = np.zeros(corrected_result.shape)
    print(corrected_result.shape)
    print(derivative_matrix.shape)

    for layer in range(corrected_result.shape[2] - 1):
        derivative_matrix[:, :, layer] = generate_derivative_matrix(corrected_result[:, :, layer], delta_x)

    print(size_changed + lower_shape)
    print(conditions["degreeM"] + 1)
    while (size_changed + lower_shape) != conditions["degreeM"] + 1:

        coefficient_matrix = create_coeff_matrix(corrected_solution_size, conditions["degreeM"])
        for t in range(corrected_result.shape[0] - 1):
            for x in range(corrected_result.shape[1] - 1):
                u_k = corrected_result[t, x, :]  # t-1 or x-1
                derivative_vector = derivative_matrix[t, x, :]
                # Runge-Kutta computations
                k1 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_vector)
                u_temp = u_k + 0.5 * delta_t * k1
                k2 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
                u_temp = u_k + 0.5 * delta_t * k2
                k3 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
                u_temp = u_k + delta_t * k3
                k4 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)

                # Update the higher degree result
                # print(k1, k2, k3, k4)
                corrected_result[t + 1, x, -1] = u_k[-1] + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        size_changed += 1

    return corrected_result


def bluff_and_fix_algorithm(conditions, correction_size):
    lower_result = compute_lower_solution(conditions)
    result = compute_higher_solution(lower_result, conditions, correction_size)
    return result

def recurssive_BNF(conditions):
    required_degree = conditions["degreeM"]
    current_condition = conditions.copy()
    current_condition["degreeM"] = 4
    result = compute_lower_solution(current_condition)
    while current_condition["degreeM"] <= required_degree:
        result = compute_higher_solution(result, current_condition, correction_size = 1)
        current_condition["degreeM"] += 1
        print("--------------------------------------------------")

    return result

# Example usage with initial conditions
example_conditions = {
    "x_start": 0,
    "x_end": 3,
    "t_start": 0,
    "t_end": 0.3,
    "num_x": 200,
    "num_t": 60,
    "degreeM": 5,
    "viscosity": 0.0
}


def revise_sequence(sequence, n):
    """
    Evaluate a Hermite polynomial sequence.

    Args:
        sequence (list): List of sequence values.
        n (int): Polynomial degree.

    Returns:
        list: List of polynomial values.
    """
    return [eval_hermite(n, x) for x in sequence]


def visualize(solution, conditions):
    random_num = [0.05]
    result_matrix = solution
    accumulated_matrix = np.zeros((result_matrix.shape[1], result_matrix.shape[0]))
    result_matrix = np.transpose(result_matrix, axes=(1, 0, 2))  # This makes it (x, t, degree)

    for j in range(result_matrix.shape[2]):
        # Assuming revise_sequence correctly matches the dimensions
        accumulated_matrix += result_matrix[:, :, j] * revise_sequence(random_num, j)[0]

    x_vals = np.linspace(conditions["x_start"], conditions["x_end"], conditions["num_x"])
    t_vals = np.linspace(conditions["t_start"], conditions["t_end"], conditions["num_t"])
    X, T = np.meshgrid(x_vals, t_vals)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Ensure the shape of accumulated_matrix matches X and T
    accumulated_matrix = np.transpose(accumulated_matrix)  # Transpose to match (t, x)
    # print(accumulated_matrix.shape)  # Should now match (num_t, num_x)
    ax.plot_surface(X, T, accumulated_matrix, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    return fig, accumulated_matrix


def animate_wave(matrix, interval=120):
    time_steps, spatial_steps = matrix.shape
    x = np.linspace(0, 3, spatial_steps)
    t = np.linspace(0, 0.3, time_steps)

    # Initialize the figure
    fig, ax = plt.subplots()

    # Initialize a line object on the plot (it will be updated later)
    line, = ax.plot(x, matrix[0])

    # Set axis labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

    # Set the y-axis limits based on your data
    ax.set_ylim(matrix.min() - matrix.mean() / 2, matrix.max() + matrix.mean() / 2)

    # Function to update the plot for each frame
    def update_plot(frame):
        line.set_ydata(matrix[frame])
        title.set_text(f'Wave Animation at Time = {t[frame]:.2f}')
        return line, title

    ani = FuncAnimation(fig, update_plot, frames=time_steps, blit=True, repeat=True, interval=interval)

    return ani

higher_degree_solution1 = recurssive_BNF(example_conditions)
higher_degree_solution2 = bluff_and_fix_algorithm(example_conditions, correction_size=1)
higher_degree_solution3 = bluff_and_fix_algorithm(example_conditions, correction_size=3)
higher_degree_solution4 = bluff_and_fix_algorithm(example_conditions, correction_size=4)


fig1, matrix1 = visualize(higher_degree_solution1, example_conditions)
fig2, matrix2 = visualize(higher_degree_solution2, example_conditions)
fig3, matrix3 = visualize(higher_degree_solution3, example_conditions)
fig4, matrix4 = visualize(higher_degree_solution4, example_conditions)
plt.figure(fig1.number)
plt.title("iterative")
plt.figure(fig2.number)
plt.title("correction_size2")

plt.figure(fig3.number)
plt.title("correction_size3")
plt.figure(fig4.number)
plt.title("correction_size4")
# animate1 = animate_wave(matrix)

degree5 = GalerkinSolver(**example_conditions)
general_solution = degree5.solve()
fig5, matrix5 = visualize(general_solution, example_conditions)
plt.figure(fig5.number)
plt.title("direct")


def error_examine(result1, result2, layer = 0):
    assert (result1.shape[2] == result2.shape[2])
    error = 0
    if layer == 0:
        for layer in range(result1.shape[2] - 1):
            error += np.linalg.norm(
                result1[:(result1.shape[0] - 1), :(result1.shape[1] - 1), layer] -
                result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer]
            ) / np.linalg.norm(result2[1:(result2.shape[0] - 1), 1:(result2.shape[1] - 1), layer])

        print(f"The error is:{error/result2.shape[2]}")

    else:
        assert (layer <= result2.shape[2])
        for layer in range(layer, layer + 1):
            error += np.linalg.norm(
                result1[:(result1.shape[0] - 1), :(result1.shape[1] - 1), layer] -
                result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer]
            ) / np.linalg.norm(result2[1:(result2.shape[0] - 1), 1:(result2.shape[1] - 1), layer])

        print(f"The error is:{error / result2.shape[2]}")
    return error/result2.shape[2]
error1 = error_examine(higher_degree_solution1, general_solution, layer=0)
error2 = error_examine(higher_degree_solution2, general_solution, layer=0)
error3 = error_examine(higher_degree_solution3, general_solution, layer=0)
error4 = error_examine(higher_degree_solution4, general_solution, layer=0)
plt.show()


