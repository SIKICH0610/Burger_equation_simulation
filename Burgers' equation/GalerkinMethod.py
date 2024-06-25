import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import eval_hermite


def create_coeff_matrix(k, degreeM):
    assert (k <= degreeM)
    result_coefficient_matrix = np.zeros((degreeM + 1, degreeM + 1))
    for i in range(0,degreeM + 1):
        for j in range(0,degreeM + 1):
            s = (i + j + k)/2
            if ((i + j + k) % 2) == 0:
                if max(i, j, k) <= s:
                    result_coefficient_matrix[i, j] = (np.math.factorial(int(i)) * np.math.factorial(
                        int(j)) /(np.math.factorial(int(s - i)) * np.math.factorial(int(s - j)) * np.math.factorial(int(s - k))))

    return result_coefficient_matrix

class GalerkinSolver:
    def __init__(self, x_start, x_end, t_start, t_end, num_x, num_t, degreeM, viscosity):
        """
        Initialize the GalerkinSolver class with the given parameters.

        Args:
            x_start (float): Starting x value for the grid.
            x_end (float): Ending x value for the grid.
            t_start (float): Starting time value for the grid.
            t_end (float): Ending time value for the grid.
            num_x (int): Number of x grid points.
            num_t (int): Number of time grid points.
            degreeM (int): Degree of expansion for the Galerkin method.
            viscosity (float): Viscosity coefficient for the problem.
        """
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.num_x = num_x
        self.num_t = num_t
        self.degreeM = degreeM
        self.viscosity = viscosity
        self.delta_x = (x_end - x_start) / num_x
        self.delta_t = (t_end - t_start) / num_t
        self.x_vals = np.linspace(x_start, x_end, num_x)
        self.t_vals = np.linspace(t_start, t_end, num_t)
        self.original_matrix = np.zeros((num_x, num_t, degreeM + 1))
        self.final_matrix = np.zeros((num_x, num_t))

    def determine_validindex(self, k):
        """
        Determine the coefficient matrix for a given index `k` and expansion degree.

        Args:
            k (int): Index to calculate the coefficient matrix for.

        Returns:
            numpy.ndarray: Coefficient matrix used for numerical calculations.
        """
        assert k <= self.degreeM
        result_coefficient_matrix = np.zeros((self.degreeM + 1, self.degreeM + 1))
        for i in range(self.degreeM + 1):
            for j in range(self.degreeM + 1):
                s = (i + j + k) / 2
                if ((i + j + k) % 2) == 0 and max(i, j, k) <= s:
                    result_coefficient_matrix[i, j] = (np.math.factorial(int(i)) * np.math.factorial(
                        int(j)) * np.math.factorial(int(k))) / \
                                                      (np.math.factorial(int(s - i)) * np.math.factorial(
                                                          int(s - j)) * np.math.factorial(
                                                          int(s - k)) * np.math.factorial(int(k)))
        return result_coefficient_matrix

    def runge_kutta_method(self, y_last, k1, k2, k3, k4):
        """
        Apply the Runge-Kutta method to integrate the solution.

        Args:
            y_last (float): Last known solution value.
            k1 (float): First RK coefficient.
            k2 (float): Second RK coefficient.
            k3 (float): Third RK coefficient.
            k4 (float): Fourth RK coefficient.

        Returns:
            float: New solution value after integration.
        """
        return y_last + self.delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def d_dx(self, vector):
        """
        Calculate the first derivative of a vector using finite differences.

        Args:
            vector (numpy.ndarray): Input vector to differentiate.

        Returns:
            numpy.ndarray: Vector of first derivatives.
        """
        result_vector = np.zeros(vector.shape[0])
        result_vector[0] = (vector[1] - vector[0]) / self.delta_x
        for i in range(1, vector.shape[0] - 1):
            result_vector[i] = (vector[i + 1] - vector[i - 1]) / (2 * self.delta_x)
        result_vector[-1] = (vector[-1] - vector[-2]) / self.delta_x
        return result_vector

    def second_d_dx(self, u):
        """
        Calculate the second derivative of a vector using finite differences.

        Args:
            u (numpy.ndarray): Input vector to differentiate.

        Returns:
            numpy.ndarray: Vector of second derivatives.
        """
        result_vec = np.zeros(u.shape[0], dtype=np.float64)
        result_vec[0] = (u[2] - 2 * u[1] + u[0])
        for i in range(1, u.shape[0] - 1):
            result_vec[i] = (u[i + 1] - 2 * u[i] + u[i - 1])
        result_vec[-1] = (u[-1] - 2 * u[-2] + u[-3])
        return result_vec / self.delta_x ** 2

    def viscosity_term(self, uslide):
        """
        Compute the viscosity term for the equation.

        Args:
            uslide (numpy.ndarray): Vector to which viscosity is applied.

        Returns:
            numpy.ndarray: Viscosity term values.
        """
        return self.viscosity * self.second_d_dx(uslide)

    def revise_sequence(self, sequence, n):
        """
        Evaluate a Hermite polynomial sequence.

        Args:
            sequence (list): List of sequence values.
            n (int): Polynomial degree.

        Returns:
            list: List of polynomial values.
        """
        return [eval_hermite(n, x) for x in sequence]

    def generate_derivative_matrix(self, input_vec_matrix):
        """
        Generate a matrix of derivatives for each column vector.

        Args:
            input_vec_matrix (numpy.ndarray): Matrix containing column vectors.

        Returns:
            numpy.ndarray: Matrix of first derivatives.
        """
        derivative_matrix = np.zeros(input_vec_matrix.shape)
        for i in range(input_vec_matrix.shape[1]):
            derivative_matrix[:, i] = self.d_dx(input_vec_matrix[:, i])
        return derivative_matrix

    def rk_helper_galerkin(self, input_vec_matrix):
        """
        Apply the Galerkin method with Runge-Kutta integration.

        Args:
            input_vec_matrix (numpy.ndarray): Input matrix representing the current state.

        Returns:
            numpy.ndarray: Resulting matrix after applying the method.
        """
        # assert input_vec_matrix.shape[1] == self.degreeM + 1
        derivative_matrix_x = self.generate_derivative_matrix(input_vec_matrix)
        result_matrix = np.zeros(input_vec_matrix.shape)

        for i in range(input_vec_matrix.shape[0]):
            u_vec_slice = input_vec_matrix[i, :]
            derivative_slice = derivative_matrix_x[i, :]
            for j in range(self.degreeM + 1):
                coefficient_matrix = self.determine_validindex(j)
                u_last = u_vec_slice[j]
                u_k = u_vec_slice.copy()
                viscosity = self.viscosity_term(input_vec_matrix[:, j])[i]

                k1 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
                u_k[j] = u_last + 0.5 * self.delta_t * k1
                k2 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
                u_k[j] = u_last + 0.5 * self.delta_t * k2
                k3 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity
                u_k[j] = u_last + self.delta_t * k3
                k4 = -np.dot(np.dot(u_k, coefficient_matrix), derivative_slice) + viscosity

                result = self.runge_kutta_method(u_last, k1, k2, k3, k4)
                result_matrix[i, j] = result

        return result_matrix

    def solve(self):
        """
        Solve the differential equation using the Galerkin and Runge-Kutta methods.

        Returns:
            numpy.ndarray: Final solution matrix after solving.
        """
        original_matrix = self.original_matrix.copy()
        original_matrix[:, 0, 1] = np.sin(self.x_vals)

        for i in range(self.num_t - 1):
            input_matrix = original_matrix[:, i, :]
            result_matrix = self.rk_helper_galerkin(input_matrix)
            original_matrix[:, i + 1, :] = result_matrix

        original_matrix = np.transpose(original_matrix,axes=(1, 0, 2))
        return original_matrix

    def visualize(self):
        random_num = [0.05]
        result_matrix = self.solve()
        accumulated_matrix = np.zeros((self.original_matrix.shape[1], self.original_matrix.shape[0]))
        for j in range(self.degreeM + 1):
            accumulated_matrix += result_matrix[:, :, j] * self.revise_sequence(random_num, j)[0]

        X, T = np.meshgrid(self.x_vals, self.t_vals)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, T, accumulated_matrix, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')
        plt.show()
        return fig, accumulated_matrix


# solver = GalerkinSolver(x_start=0, x_end=3, t_start=0, t_end=0.2, num_x=200, num_t=60, degreeM=5, viscosity= 0)
# result = solver.solve()
# solver.visualize()

# print(create_coeff_matrix(5,5))
