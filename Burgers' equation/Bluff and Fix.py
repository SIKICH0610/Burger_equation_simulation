import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite
from Galerkin_Solver import GalerkinSolver, create_coeff_matrix
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


def compute_lower_solution(conditions):
    """
    Compute solutions for degree M-1 using a Galerkin solver.

    Args:
        conditions (dict): Dictionary containing solver conditions.

    Returns:
        np.ndarray: Lower degree solution result.
    """
    assert len(conditions) == 8, "Conditions must include all necessary parameters."

    lower_degree_conditions = conditions.copy()
    lower_degree_conditions["degreeM"] -= 1
    lower_solver = GalerkinSolver(**lower_degree_conditions)
    lower_result = lower_solver.solve()

    return lower_result


def compute_higher_solution(lower_result, conditions, correction_size):
    """
    Adjusts the solution to a higher degree by applying corrections iteratively.

    Args:
        lower_result (np.ndarray): The lower degree solution result.
        conditions (dict): Dictionary containing solver conditions.
        correction_size (int): Size of the correction to apply.

    Returns:
        np.ndarray: Corrected higher degree solution result.
    """
    corrected_solution_size = conditions["degreeM"] - correction_size
    corrected_result = lower_result[:, :, :corrected_solution_size + 1]
    lower_shape = corrected_result.shape[2]

    delta_t = (conditions['t_end'] - conditions['t_start']) / conditions['num_t']
    delta_x = (conditions['x_end'] - conditions['x_start']) / conditions['num_x']
    size_changed = 0
    shape = (corrected_result.shape[0], corrected_result.shape[1], correction_size)
    corrected_result = np.concatenate((corrected_result, np.zeros(shape)), axis=2)
    derivative_matrix = np.zeros(corrected_result.shape)

    for layer in range(corrected_result.shape[2]):
        derivative_matrix[:, :, layer] = generate_derivative_matrix(corrected_result[:, :, layer], delta_x)

    while (size_changed + lower_shape) != conditions["degreeM"] + 1:
        coefficient_matrix = create_coeff_matrix(corrected_solution_size, conditions["degreeM"])
        for t in range(corrected_result.shape[0] - 1):
            for x in range(corrected_result.shape[1] - 1):
                u_k = corrected_result[t, x, :]
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
                corrected_result[t + 1, x, -1] = u_k[-1] + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        size_changed += 1

    return corrected_result


def compute_specific_layer(previous_result, conditions, specific_layer):
    """
    Computes a specific layer of the solution matrix.

    Args:
        previous_result (np.ndarray): The result before the specific layer.
        conditions (dict): Dictionary containing solver conditions.
        specific_layer (int): The specific layer to compute.

    Returns:
        np.ndarray: Computed specific layer result.
    """
    delta_t = (conditions['t_end'] - conditions['t_start']) / conditions['num_t']
    delta_x = (conditions['x_end'] - conditions['x_start']) / conditions['num_x']
    derivative_matrix = np.zeros(previous_result.shape)

    for layer in range(previous_result.shape[2]):
        derivative_matrix[:, :, layer] = generate_derivative_matrix(previous_result[:, :, layer], delta_x)

    coefficient_matrix = create_coeff_matrix(specific_layer, conditions["degreeM"])

    for t in range(previous_result.shape[0] - 1):
        for x in range(previous_result.shape[1]):
            u_k = previous_result[t, x, :]
            derivative_vector = derivative_matrix[t, x, :]
            # Runge-Kutta computations
            k1 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_vector)
            u_temp = u_k + 0.5 * delta_t * k1
            k2 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
            u_temp = u_k + 0.5 * delta_t * k2
            k3 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)
            u_temp = u_k + delta_t * k3
            k4 = -np.dot(np.dot(u_temp, coefficient_matrix), derivative_vector)

            previous_result[t + 1, x, -1] = u_k[-1] + (delta_t / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return previous_result[:, :, specific_layer]


def bluff_and_fix_algorithm(conditions, correction_size):
    """
    Combines compute_lower_solution and compute_higher_solution to produce the final result.

    Args:
        conditions (dict): Dictionary containing solver conditions.
        correction_size (int): Size of the correction to apply.

    Returns:
        np.ndarray: Final solution result.
    """
    lower_result = compute_lower_solution(conditions)
    result = compute_higher_solution(lower_result, conditions, correction_size)
    return result


def random_BNF(conditions, correction_size):
    """
    Randomly selects layers to compute using the Bluff and Fix algorithm.

    Args:
        conditions (dict): Dictionary containing solver conditions.
        correction_size (int): Size of the correction to apply.

    Returns:
        np.ndarray: Final solution result with random layers.
    """
    correction_size -= 1
    required_degree = conditions["degreeM"]
    result = compute_lower_solution(conditions)
    result = np.concatenate((result, np.zeros((result.shape[0], result.shape[1], 1))), axis=2)
    random_layer = random.sample(range(2, conditions["degreeM"]), correction_size)
    random_layer.append(required_degree)
    # random_layer = [2,5]
    print(random_layer)


    for layer in random_layer:
        result[:, :, layer] = 0

    for item in random_layer:
        result[:, :, item] = compute_specific_layer(result, conditions, item)

    return result


def recurssive_BNF(conditions, correction_size, starting_layer):
    """
    Recursively applies the Bluff and Fix algorithm to compute solutions for increasing degrees.

    Args:
        conditions (dict): Dictionary containing solver conditions.
        correction_size (int): Size of the correction to apply for each step.
        starting_layer (int): Initial degree to start computation from.

    Returns:
        np.ndarray: Final solution result.
    """
    required_degree = conditions["degreeM"]
    current_condition = conditions.copy()
    current_condition["degreeM"] = starting_layer + 1
    # This + 1 here is serving to make the starting_layer to be correctly represented during computing, since the
    # compute_lower_solution method has - 1 for the condition
    result = compute_lower_solution(current_condition)

    while current_condition["degreeM"] <= required_degree:
        result = compute_higher_solution(result, current_condition, correction_size = correction_size)
        current_condition["degreeM"] += correction_size

    return result


def revise_sequence(sequence, n):
    """
    Evaluate a Hermite polynomial sequence, using to generate true solution and visualization

    Args:
        sequence (list): List of sequence values.
        n (int): Polynomial degree.

    Returns:
        list: List of polynomial values.
    """
    return [eval_hermite(n, x) for x in sequence]


def visualize(solution, conditions, layer_range=0, vis_type='accumulate'):
    """
    Visualizes the solution matrix as a 3D surface plot.

    Args:
        solution (np.ndarray): The solution matrix.
        conditions (dict): Dictionary containing solver conditions.
        layer_range (int): Which u_int to be visualized.
        vis_type (str): Type of visualization, either 'single' to show a single layer, 'accumulate' to accumulate all
                        layers before given layer_range, or 'full solution' to show accumulated all layers.

    Returns:
        tuple: Figure object and accumulated matrix.
    """
    result_matrix = np.transpose(solution, axes=(1, 0, 2))  # (x, t, degree)
    x_vals = np.linspace(conditions["x_start"], conditions["x_end"], conditions["num_x"])
    t_vals = np.linspace(conditions["t_start"], conditions["t_end"], conditions["num_t"])
    X, T = np.meshgrid(x_vals, t_vals)
    title = ""

    if vis_type == 'single':
        accumulated_matrix = result_matrix[:, :, layer_range]
        title = f"Single Layer Plot of Layer {layer_range}"
    elif vis_type == 'accumulate':
        random_num = [0.05]
        accumulated_matrix = np.zeros((result_matrix.shape[0], result_matrix.shape[1]))
        for j in range(layer_range):
            accumulated_matrix += result_matrix[:, :, j] * revise_sequence(random_num, j)[0]
        title = f"Accumulated Plot Up to Layer {layer_range}"
    elif vis_type == 'full_solution':
        random_num = [0.05]
        accumulated_matrix = np.zeros((result_matrix.shape[0], result_matrix.shape[1]))
        for j in range(result_matrix.shape[2]):
            accumulated_matrix += result_matrix[:, :, j] * revise_sequence(random_num, j)[0]
        title = "Full Solution Accumulated Plot"
    else:
        raise ValueError("vis_type must be either 'single', 'accumulate' or 'full solution'")

    accumulated_matrix = np.transpose(accumulated_matrix)  # Transpose to match (t, x)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, accumulated_matrix, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u')
    ax.set_title(title)

    return fig, accumulated_matrix


def animate_wave(matrix, interval=120):
    """
    Creates an animation of the wave solution over time.

    Args:
        matrix (np.ndarray): The solution matrix.
        interval (int, optional): Time interval between frames. Defaults to 120.

    Returns:
        FuncAnimation: Animation object.
    """
    time_steps, spatial_steps = matrix.shape
    x = np.linspace(0, 3, spatial_steps)
    t = np.linspace(0, 0.3, time_steps)

    fig, ax = plt.subplots()
    line, = ax.plot(x, matrix[0])
    ax.set_xlabel('x')
    ax.set_ylabel('u')
    title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')
    ax.set_ylim(matrix.min() - matrix.mean() / 2, matrix.max() + matrix.mean() / 2)

    def update_plot(frame):
        line.set_ydata(matrix[frame])
        title.set_text(f'Wave Animation at Time = {t[frame]:.2f}')
        return line, title

    ani = FuncAnimation(fig, update_plot, frames=time_steps, blit=True, repeat=True, interval=interval)
    return ani


def normalize_solution(solution):
    """
    Normalizes the solution to a range [0, 1].

    Args:
        solution (np.ndarray): The solution matrix to be normalized.

    Returns:
        np.ndarray: Normalized solution matrix.
    """
    min_val = np.min(solution)
    max_val = np.max(solution)
    normalized_solution = (solution - min_val) / (max_val - min_val)
    return normalized_solution


def error_examine(result1, result2, layer=0, method='norm'):
    """
    Examines the error between two result matrices.

    Args:
        result1 (np.ndarray): The first result matrix.
        result2 (np.ndarray): The second result matrix.
        layer (int, optional): Specific layer to examine. Defaults to 0.
        method (str, optional): Error computation method, 'norm' for normalization, 'shape' for shape comparison.

    Returns:
        float: Computed error.
    """
    assert result1.shape[2] == result2.shape[2], "The number of layers in both results must be the same."

    if method == 'norm':
        # Normal error computation
        error = 0
        if layer == 0:
            for layer in range(result1.shape[2] - 1):
                error += np.linalg.norm(
                    result1[:(result1.shape[0] - 1), :(result1.shape[1] - 1), layer] -
                    result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer]
                ) / np.linalg.norm(result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer])

            error /= result2.shape[2]
        else:
            assert layer <= result2.shape[2], "Layer index out of bounds."
            error += np.linalg.norm(
                result1[:(result1.shape[0] - 1), :(result1.shape[1] - 1), layer] -
                result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer]
            ) / np.linalg.norm(result2[:(result2.shape[0] - 1), :(result2.shape[1] - 1), layer])
            error /= result2.shape[2]

    elif method == 'shape':
        # Shape comparison error computation
        normalized_result1 = normalize_solution(result1)
        normalized_result2 = normalize_solution(result2)
        error = 0
        if layer == 0:
            for layer in range(normalized_result1.shape[2] - 1):
                error += np.linalg.norm(
                    normalized_result1[:(normalized_result1.shape[0] - 1), :(normalized_result1.shape[1] - 1), layer] -
                    normalized_result2[:(normalized_result2.shape[0] - 1), :(normalized_result2.shape[1] - 1), layer]
                ) / np.linalg.norm(
                    normalized_result2[:(normalized_result2.shape[0] - 1), :(normalized_result2.shape[1] - 1), layer])

            error /= normalized_result2.shape[2]
        else:
            assert layer <= normalized_result2.shape[2], "Layer index out of bounds."
            error += np.linalg.norm(
                normalized_result1[:(normalized_result1.shape[0] - 1), :(normalized_result1.shape[1] - 1), layer] -
                normalized_result2[:(normalized_result2.shape[0] - 1), :(normalized_result2.shape[1] - 1), layer]
            ) / np.linalg.norm(
                normalized_result2[:(normalized_result2.shape[0] - 1), :(normalized_result2.shape[1] - 1), layer])
            error /= normalized_result2.shape[2]

    else:
        raise ValueError("method must be either 'norm' or 'shape'")

    print(f"The error is: {error}")
    return error


example_conditions = {
    "x_start": 0,
    "x_end": 3,
    "t_start": 0,
    "t_end": 0.2,
    "num_x": 201,
    "num_t": 61,
    "degreeM": 5,
    "viscosity": 0.0
}

# higher_degree_solution1 = recurssive_BNF(example_conditions, correction_size=1, starting_layer= 2)
# higher_degree_solution2 = bluff_and_fix_algorithm(example_conditions, correction_size=1)
# higher_degree_solution3 = bluff_and_fix_algorithm(example_conditions, correction_size=3)
# higher_degree_solution4 = bluff_and_fix_algorithm(example_conditions, correction_size=4)
higher_degree_solution5 = random_BNF(example_conditions, correction_size= 2)


# fig1, matrix1 = visualize(higher_degree_solution1, example_conditions)
# fig2, matrix2 = visualize(higher_degree_solution2, example_conditions)
# fig3, matrix3 = visualize(higher_degree_solution3, example_conditions)
# fig4, matrix4 = visualize(higher_degree_solution4, example_conditions)
fig5, matrix5 = visualize(higher_degree_solution5, example_conditions, vis_type= "full_solution")
# plt.figure(fig1.number)
# plt.title("iterative")
# plt.figure(fig2.number)
# plt.title("correction_size2")
#
# plt.figure(fig3.number)
# plt.title("correction_size3")
# plt.figure(fig4.number)
# plt.title("correction_size4")
# animate1 = animate_wave(matrix)

degree5 = GalerkinSolver(**example_conditions)
general_solution = degree5.solve()
fig6, matrix6 = visualize(general_solution, example_conditions, layer_range= 4, vis_type= "single")
plt.figure(fig6.number)
plt.title("direct")

# error1 = error_examine(higher_degree_solution1, general_solution, layer=0)
# error2 = error_examine(higher_degree_solution2, general_solution, layer=0)
# error3 = error_examine(higher_degree_solution3, general_solution, layer=0)
# error4 = error_examine(higher_degree_solution4, general_solution, layer=0)
error5 = error_examine(higher_degree_solution5, general_solution, layer=0, method = "norm")
error5 = error_examine(higher_degree_solution5, general_solution, layer=0, method = "shape")
plt.show()


