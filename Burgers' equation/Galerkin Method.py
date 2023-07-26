import numpy as np
import matplotlib.pyplot as plt

def Runge_Kutta_method(y_last, k1, k2, k3, k4, delta_t):
    y_new = y_last + delta_t * (k1 + 2 * k2 + 2 * k3 + k4)/6
    return y_new

def k_computing_central(pre_vector0, pre_vector1, delta_x, delta_t, v):
    length = pre_vector0.shape[0]
    result_vector0 = np.zeros(pre_vector0.shape[0])
    result_vector1 = np.zeros(pre_vector1.shape[0])
    # This part computing for the first component of the vector
    index0_0_k1 = -pre_vector0[0] * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector1[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                                    pre_vector0[2] - 2 * pre_vector0[1] + pre_vector0[0]) / delta_x ** 2
    index1_0_k1 = -pre_vector1[0] * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector0[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                                    pre_vector1[2] - 2 * pre_vector1[1] + pre_vector1[0]) / delta_x ** 2


    index0_0_k2 = -(pre_vector0[0] + (index0_0_k1 * delta_x) / 2) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector1[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                    pre_vector0[2] - 2 * pre_vector0[1] + pre_vector0[0]) / delta_x ** 2
    index1_0_k2 = -(pre_vector1[0] + (index1_0_k1 * delta_x) / 2) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector0[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                    pre_vector1[2] - 2 * pre_vector1[1] + pre_vector1[0]) / delta_x ** 2


    index0_0_k3 = -(pre_vector0[0] + (index0_0_k2 * delta_x) / 2) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector1[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                    pre_vector0[2] - 2 * pre_vector0[1] + pre_vector0[0]) / delta_x ** 2
    index1_0_k3 = -(pre_vector1[0] + (index1_0_k2 * delta_x) / 2) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector0[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                    pre_vector1[2] - 2 * pre_vector1[1] + pre_vector1[0]) / delta_x ** 2


    index0_0_k4 = -(pre_vector0[0] + index0_0_k3 * delta_x) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector1[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                pre_vector0[2] - 2 * pre_vector0[1] + pre_vector0[0]) / delta_x ** 2
    index1_0_k4 = -(pre_vector1[0] + index1_0_k3 * delta_x) * (pre_vector0[1] - pre_vector0[0]) / delta_x - pre_vector0[0] * (pre_vector1[1] - pre_vector1[0]) / delta_x + v * (
                pre_vector1[2] - 2 * pre_vector1[1] + pre_vector1[0]) / delta_x ** 2



    result_vector0[0] = Runge_Kutta_method(pre_vector0[0], index0_0_k1, index0_0_k2, index0_0_k3, index0_0_k4, delta_t)
    result_vector1[0] = Runge_Kutta_method(pre_vector1[0], index1_0_k1, index1_0_k2, index1_0_k3, index1_0_k4, delta_t)

    for i in range(1,length-1):
        # This part computing for components that from second to the one before the last
        y_0, y_1 = pre_vector0[i],pre_vector1[i]

        component0_0 = (y_1 * pre_vector1[i+1] - pre_vector1[i-1]) / (2 * delta_x)
        component0_1 = v * (pre_vector0[i + 1] - 2*pre_vector0[i] + pre_vector0[i - 1])/ delta_x**2
        component1_0 = (y_0 * pre_vector1[i+1] - pre_vector1[i-1]) / (2 * delta_x)
        component1_1 = v * (pre_vector1[i + 1] - 2*pre_vector1[i] + pre_vector1[i - 1])/ delta_x**2

        v0_k1 = -y_0 * (pre_vector0[i+1] - pre_vector0[i-1]) / (2 * delta_x) - component0_0 + component0_1
        v1_k1 = -y_1 * (pre_vector0[i+1] - pre_vector0[i-1]) / (2 * delta_x) - component1_0 + component1_1

        v0_k2 = -(y_0 + (v0_k1 * delta_x) / 2) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1
        v1_k2 = -(y_1 + (v1_k1 * delta_x) / 2) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1

        v0_k3 = -(y_0 + (v0_k2 * delta_x) / 2) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1
        v1_k3 = -(y_1 + (v1_k2 * delta_x) / 2) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1


        v0_k4 = -(y_0 + (v0_k3 * delta_x)) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1
        v1_k4 = -(y_1 + (v1_k3 * delta_x)) * (pre_vector0[i + 1] - pre_vector0[i - 1]) / (2 * delta_x) - component0_0 + component0_1


        result_vector0[i] = Runge_Kutta_method(y_0, v0_k1, v0_k2, v0_k3, v0_k4, delta_t)
        result_vector1[i] = Runge_Kutta_method(y_1, v1_k1, v1_k2, v1_k3, v1_k4, delta_t)


    # This part compute for the last component
    index0_0_k1 = -pre_vector0[-1] * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector1[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                                pre_vector0[-3] - 2 * pre_vector0[-2] + pre_vector0[-1]) / delta_x ** 2
    index1_0_k1 = -pre_vector1[-1] * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector0[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                                pre_vector1[-3] - 2 * pre_vector1[-2] + pre_vector1[-1]) / delta_x ** 2

    index0_0_k2 = -(pre_vector0[-1] + (index0_0_k1 * delta_x) / 2) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector1[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector0[-3] - 2 * pre_vector0[-2] + pre_vector0[-1]) / delta_x ** 2
    index1_0_k2 = -(pre_vector1[-1] + (index1_0_k1 * delta_x) / 2) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector0[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector1[-3] - 2 * pre_vector1[-2] + pre_vector1[-1]) / delta_x ** 2

    index0_0_k3 = -(pre_vector0[-1] + (index0_0_k2 * delta_x) / 2) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector1[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector0[-3] - 2 * pre_vector0[-2] + pre_vector0[-1]) / delta_x ** 2
    index1_0_k3 = -(pre_vector1[-1] + (index1_0_k2 * delta_x) / 2) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector0[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector1[-3] - 2 * pre_vector1[-2] + pre_vector1[-1]) / delta_x ** 2

    index0_0_k4 = -(pre_vector0[-1] + index0_0_k3 * delta_x) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector1[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector0[-3] - 2 * pre_vector0[-2] + pre_vector0[-1]) / delta_x ** 2
    index1_0_k4 = -(pre_vector1[-1] + index1_0_k3 * delta_x) * (pre_vector0[-2] - pre_vector0[-1]) / delta_x - pre_vector0[-1] * (pre_vector1[-2] - pre_vector1[-1]) / delta_x + v * (
                pre_vector1[-3] - 2 * pre_vector1[-2] + pre_vector1[-1]) / delta_x ** 2

    result_vector0[-1] = Runge_Kutta_method(pre_vector0[-1], index0_0_k1, index0_0_k2, index0_0_k3, index0_0_k4, delta_t)
    result_vector1[-1] = Runge_Kutta_method(pre_vector1[-1], index1_0_k1, index1_0_k2, index1_0_k3, index1_0_k4, delta_t)

    return result_vector0,result_vector1

def Galerkincomputing(Grid2D0, Grid2D1, x_start, x_end, t_start, t_end, v):
    # Galerkin projection method with M = 1
    gridsize = Grid2D0.shape
    x_gridsize = gridsize[1]
    t_gridsize = gridsize[0]
    delta_x = (x_end - x_start)/x_gridsize
    delta_t = (t_end - t_start)/t_gridsize
    for i in range(0, t_gridsize - 1): # Computing for u_0 amd u_1
        matrix_slice0 = Grid2D0[i, :]
        matrix_slice1 = Grid2D1[i, :]
        Grid2D0[i+1,:], Grid2D1[i+1,:] =k_computing_central(matrix_slice0,matrix_slice1,delta_x, delta_t,v)

    return Grid2D0, Grid2D1

x_start, x_end = 0, 3
t_start, t_end = 0, 0.2
num_points_x = 60 # Number of grid points in the x direction
num_points_t = 200  # Number of grid points in the t direction
x_vals = np.linspace(x_start, x_end, num_points_x)
t_vals = np.linspace(t_start, t_end, num_points_t)
u0 = np.zeros((num_points_t, num_points_x), dtype = np.float64)
u1 = np.zeros((num_points_t, num_points_x), dtype = np.float64)
u1[0, :] = np.sin(x_vals)
k0, k1 = Galerkincomputing(u0, u1, x_start, x_end, t_start, t_end, 0.05)
random_num = np.random.rand(1)
result_matrix = k0 + random_num * k1

X, T = np.meshgrid(x_vals, t_vals)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, result_matrix, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()
