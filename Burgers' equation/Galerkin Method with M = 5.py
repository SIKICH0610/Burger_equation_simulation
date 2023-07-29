import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite

def determine_validindex(size1,size2,k):
    for i in range(0,size1 + 1):
        for j in range(0,size2 + 1):
            s = (i + j + k)/2
            if ((i + j + k) % 2) == 0:
                if max(i, j, k) <= s:
                    print(i,j)
                    print((np.math.factorial(i) * np.math.factorial(j) * np.math.factorial(k))/(np.math.factorial(s - i) * np.math.factorial(s - j) * np.math.factorial(s - k)))
                    print("----------------")
    print("*********************")
    return None

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

def d_d0_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
        # ALL u here is the sliced vector from the matrix
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -u0 * du0_dx - u1 * du1_dx - 2 * u2 * du2_dx - 6 * u3 * du3_dx \
                    - 24 * u4 * du4_dx - 36 * u5 * du5_dx + v * Second_d_dx(u0, delta_x)
    return result_vector / np.math.factorial(0)

def d_d1_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -u0 * du1_dx - u1 * du0_dx - 2 * u1 * du2_dx - 2 *u2 * du1_dx - 6 * u2 * du3_dx - \
                    6 * u3 * du2_dx - 24 * u3 * du4_dx - 24 * u4 * du3_dx * - 120 * u4 * du5_dx - \
                    120 * u5 * du4_dx
    return  (result_vector / np.math.factorial(1)) + v * Second_d_dx(u0, delta_x)

def d_d2_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -2 * u0 * du2_dx - 2 * u1 * du1_dx - 6 * u1 * du3_dx \
                    - 2 * u2 * du0_dx - 8 * u2 * du2_dx - \
                    24 * u2 * du4_dx - 6 * u3 * du1_dx - 36 * u3 * du3_dx * - 120 * u3 * du5_dx - \
                    24 * u4 * du2_dx - 192 * u4 * du4_dx - 120 * u5 * du5_dx - 1200 * u5 * du5_dx
    return  (result_vector / np.math.factorial(2)) + v * Second_d_dx(u0, delta_x)

def d_d3_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -6 * u0 * du3_dx - 6 * u1 * du2_dx - 24 * u1 * du4_dx - 6 *u2 * du1_dx - 36 * u2 * du3_dx - \
                    120 * u2 * du5_dx - 6 * u3 * du0_dx - 36 * u3 * du2_dx * - 216 * u3 * du4_dx - \
                    24 * u4 * du1_dx - 216 * u4 * du3_dx - 1440 * u4 * du5_dx - 120 * u5 * du2_dx -\
                    1440 * u5 * du4_dx
    return  (result_vector / np.math.factorial(3)) + v * Second_d_dx(u0, delta_x)

def d_d4_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -24 * u0 * du4_dx - 24 * u1 * du3_dx - 120 * u1 * du5_dx - 24 *u2 * du2_dx - 192 * u2 * du4_dx - \
                    24 * u3 * du1_dx - 216 * u3 * du3_dx - 1440 * u3 * du5_dx * - 24 * u4 * du0_dx - \
                    192 * u4 * du2_dx - 1728 * u4 * du4_dx - 120 * u5 * du1_dx - 1440 * u5 * du3_dx - \
                    14400 * u5 * du5_dx
    return  (result_vector / np.math.factorial(4)) + v * Second_d_dx(u0, delta_x)

def d_d5_dt(u0, u1, u2, u3, u4, u5, delta_x, v):
    du0_dx = d_dx(u0, delta_x)
    du1_dx = d_dx(u1, delta_x)
    du2_dx = d_dx(u2, delta_x)
    du3_dx = d_dx(u3, delta_x)
    du4_dx = d_dx(u4, delta_x)
    du5_dx = d_dx(u5, delta_x)
    result_vector = -120 * u0 * du5_dx - 120 * u1 * du4_dx - 120 * u2 * du3_dx - 1200 * u2 * du5_dx - 120 * u3 * du2_dx - \
                    1440 * u3 * du4_dx - 120 * u4 * du1_dx - 1440 * u4 * du3_dx * - 14400 * u4 * du5_dx - \
                    120 * u5 * du0_dx - 1200 * u5 * du2_dx - 14400 * u5 * du4_dx
    return  (result_vector / np.math.factorial(5)) + v * Second_d_dx(u0, delta_x)

def RK_helper_Galerkin(u0, u1, u2, u3, u4, u5, delta_x, delta_t, v):
    k1_0, k1_1, k1_2, k1_3, k1_4, k1_5 = d_d0_dt(u0,u1,u2,u3,u4,u5,delta_x, v), d_d1_dt(u0,u1,u2,u3,u4,u5,delta_x, v),\
                                         d_d2_dt(u0,u1,u2,u3,u4,u5,delta_x, v), d_d3_dt(u0,u1,u2,u3,u4,u5,delta_x, v),\
                                         d_d4_dt(u0,u1,u2,u3,u4,u5,delta_x, v), d_d5_dt(u0,u1,u2,u3,u4,u5,delta_x, v)
    k2_0, k2_1, k2_2, k2_3, k2_4, k2_5 = d_d0_dt((u0 + 0.5 * delta_t * k1_0),u1,u2,u3,u4,u5,delta_x, v),\
                                         d_d1_dt(u0,(u1 + 0.5 * delta_t * k1_1),u2,u3,u4,u5,delta_x, v),\
                                         d_d2_dt(u0,u1,(u2 + 0.5 * delta_t * k1_2),u3,u4,u5,delta_x, v),\
                                         d_d3_dt(u0,u1,u2,(u3 + 0.5 * delta_t * k1_3),u4,u5,delta_x, v),\
                                         d_d4_dt(u0,u1,u2,u3,(u4 + 0.5 * delta_t * k1_4),u5,delta_x, v),\
                                         d_d5_dt(u0,u1,u2,u3,u4,(u5 + 0.5 * delta_t * k1_5),delta_x, v)
    k3_0, k3_1, k3_2, k3_3, k3_4, k3_5 = d_d0_dt((u0 + 0.5 * delta_t * k2_0),u1,u2,u3,u4,u5,delta_x, v),\
                                         d_d1_dt(u0,(u1 + 0.5 * delta_t * k2_1),u2,u3,u4,u5,delta_x, v),\
                                         d_d2_dt(u0,u1,(u2 + 0.5 * delta_t * k2_2),u3,u4,u5,delta_x, v),\
                                         d_d3_dt(u0,u1,u2,(u3 + 0.5 * delta_t * k2_3),u4,u5,delta_x, v),\
                                         d_d4_dt(u0,u1,u2,u3,(u4 + 0.5 * delta_t * k2_4),u5,delta_x, v),\
                                         d_d5_dt(u0,u1,u2,u3,u4,(u5 + 0.5 * delta_t * k2_5),delta_x, v)
    k4_0, k4_1, k4_2, k4_3, k4_4, k4_5 = d_d0_dt((u0 + delta_t * k3_0),u1,u2,u3,u4,u5,delta_x, v),\
                                         d_d1_dt(u0,(u1 + delta_t * k3_1),u2,u3,u4,u5,delta_x, v),\
                                         d_d2_dt(u0,u1,(u2 + delta_t * k3_2),u3,u4,u5,delta_x, v),\
                                         d_d3_dt(u0,u1,u2,(u3 + delta_t * k3_3),u4,u5,delta_x, v),\
                                         d_d4_dt(u0,u1,u2,u3,(u4 + delta_t * k3_4),u5,delta_x, v),\
                                         d_d5_dt(u0,u1,u2,u3,u4,(u5 + delta_t * k3_5),delta_x, v)
    result_u0 = Runge_Kutta_method(u0, k1_0, k2_0, k3_0, k4_0, delta_t)
    result_u1 = Runge_Kutta_method(u1, k1_1, k2_1, k3_1, k4_1, delta_t)
    result_u2 = Runge_Kutta_method(u2, k1_2, k2_2, k3_2, k4_2, delta_t)
    result_u3 = Runge_Kutta_method(u3, k1_3, k2_3, k3_3, k4_3,- delta_t)
    result_u4 = Runge_Kutta_method(u4, k1_4, k2_4, k3_4, k4_4, delta_t)
    result_u5 = Runge_Kutta_method(u5, k1_5, k2_5, k3_5, k4_5, delta_t)
    return result_u0, result_u1, result_u2, result_u3, result_u4, result_u5

def Galerkincomputing(u0, u1, u2, u3, u4, u5, x_start, x_end, t_start, t_end, v):
    # Galerkin projection method with M = 1
    gridsize = u0.shape
    x_gridsize = gridsize[1]
    t_gridsize = gridsize[0]
    delta_x = (x_end - x_start)/x_gridsize
    delta_t = (t_end - t_start)/t_gridsize
    for i in range(0, t_gridsize - 1): # Computing for u_0 amd u_1
        matrix_slice0 = u0[i, :]
        matrix_slice1 = u1[i, :]
        matrix_slice2 = u2[i, :]
        matrix_slice3 = u3[i, :]
        matrix_slice4 = u4[i, :]
        matrix_slice5 = u5[i, :]
        u0[i + 1, :], u1[i + 1, :], u2[i + 1, :], u3[i + 1, :], u4[i + 1, :], u5[i + 1, :] =\
            RK_helper_Galerkin(matrix_slice0, matrix_slice1, matrix_slice2, matrix_slice3, matrix_slice4, matrix_slice5,
                               delta_x, delta_t, v)


    return u0, u1, u2, u3, u4, u5

def revise_sequence(sequence, n):
    revised_sequence = [eval_hermite(n, x) for x in sequence]
    return revised_sequence

x_start, x_end = 0, 3
t_start, t_end = 0, 0.2
num_points_x = 60 # Number of grid points in the x direction
num_points_t = 200  # Number of grid points in the t direction
x_vals = np.linspace(x_start, x_end, num_points_x)
t_vals = np.linspace(t_start, t_end, num_points_t)
u0 = np.zeros((num_points_t, num_points_x))
u1 = np.zeros((num_points_t, num_points_x))
u2 = np.zeros((num_points_t, num_points_x))
u3 = np.zeros((num_points_t, num_points_x))
u4 = np.zeros((num_points_t, num_points_x))
u5 = np.zeros((num_points_t, num_points_x))
u1[0, :] = np.sin(x_vals)
u0, u1, u2, u3, u4, u5 = Galerkincomputing(u0, u1, u2, u3, u4, u5, x_start, x_end, t_start, t_end, 0.9)
result_matrix = u0 + revise_sequence(np.random.normal(0,1,1),1) * u1 + revise_sequence(np.random.normal(0,1,1),2) * u2 +\
                revise_sequence(np.random.normal(0,1,1),3) * u3 + revise_sequence(np.random.normal(0,1,1),4) * u4 +\
                revise_sequence(np.random.normal(0,1,1),4) * u4 + revise_sequence(np.random.normal(0,1,1),5) * u5
X, T = np.meshgrid(x_vals, t_vals)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, result_matrix, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')
plt.show()