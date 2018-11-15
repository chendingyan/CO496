import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


def grad_f1(x):
    x = np.array([[x[0]], [x[1]]])
    matrix = np.array([[8, -2], [-2, 8]])
    matrix_1 = np.array([[1], [1]])
    return (np.matmul(matrix, x)- matrix_1)

def grad_f2(x):
    a = np.array([[1],[0]])
    b = np.array([[0],[-1]])
    matrix = np.array([[6, -2], [-2, 6]])
    cos = np.cos(np.matmul(np.transpose(x-a), x-a))
    return (2*cos*(x-a)+ np.matmul(matrix, x-b))

def f2(x):
    a = np.array([[1],[0]])
    b = np.array([[0],[-1]])
    matrix = np.array([[3, -1], [-1, 3]])
    return np.sin(np.matmul(np.transpose(x-a), x-a))+ np.matmul(np.matmul(np.transpose(x-b), matrix), x-b)

def f3(x):
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    matrix = np.array([[3, -1], [-1, 3]])
    return 1-(np.exp(-np.matmul(np.transpose(x-a), x-a)) + np.exp(-np.matmul(np.matmul(np.transpose(x-b), matrix), x-b))- 0.1 * np.log(np.linalg.det( 0.01* np.identity(2) + np.matmul(x, np.transpose(x)))) )


x = np.array([[1], [-1]], dtype = float)
# print(grad_f1(x))
# print(grad_f2(x))
#
# gradient_f2 = grad(f2)
# print(gradient_f2(np.array([[1],[-1]])))

# grad_f3 = grad(f3)
# print(f3(x))
# print(grad_f3(x))




def plot_contour(delta, xrange1, xrange2, yrange1, yrange2, function):
    x_axis = np.arange(xrange1, xrange2, delta)
    y_axis = np.arange(xrange1, xrange2, delta)
    X, Y = np.meshgrid(x_axis, y_axis)
    Z = np.zeros(X.shape)
    for x1 in range(0, Z.shape[1]):
        for y1 in range(0, Z.shape[0]):
            x = np.array([[x_axis[x1]], [y_axis[y1]]], dtype=float)
            z = function(x)
            Z[y1][x1] = z
    return X, Y, Z


def gradient_descent(step_size, iterations, functions, start_point):
    x_points = np.zeros((1, iterations))
    y_points = np.zeros((1, iterations))
    for i in range(0, iterations):
        x_points[0][i] = start_point[0][0]
        y_points[0][i] = start_point[1][0]
        start_point -= step_size * functions(start_point)
    return x_points, y_points






# fig = plt.subplot(211)
fig = plt.figure(1)
X, Y, Z = plot_contour(0.025, -1, 1.5, -2, 0, f2)
CS = plt.contour(X, Y, Z)
x_points, y_points = gradient_descent(0.1, 50, grad_f2, x)
plt.scatter(x_points, y_points)
# # # ax.clabel(CS, inline=1, fontsize=10)

# x = np.array([[1], [-1]], dtype = float)
# print(x)
# fig2 = plt.subplot(212)
# X, Y, Z = plot_contour(0.025, -0.5, 1.5, -2, 1, f3)
# CS = plt.contour(X, Y, Z)
# x_points, y_points = gradient_descent(0.1, 50, grad_f3, x)
# plt.scatter(x_points, y_points)
# fig2.set_title('gradient descent of f3')
plt.show()

