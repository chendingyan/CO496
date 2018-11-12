
import numpy as np
import matplotlib.pyplot as plt


N = 25
X = np.reshape(np.linspace(0, 0.9, N), (N,1))
Y = np.cos(10*X**2) + 0.1* np.sin(100*X)

def getPhi(feature, order):
    phi = np.zeros((feature.shape[0], order + 1))
    for i in range(feature.shape[0]):
        for j in range(0, order + 1):
            phi[i][j] = np.power(feature[i], j)
    return phi

def lml(alpha, beta, Phi, Y):
    conv = alpha* np.dot(Phi, np.transpose(Phi)) + beta * np.identity(Phi.shape[0])
    t1 = (2 * np.pi)**(-Phi.shape[0]/2.0)* np.linalg.det(conv) **(-0.5)
    t2 = np.exp((-0.5)*(np.dot(np.dot(np.transpose(Y), np.linalg.inv(conv)), Y)))
    log_ml = np.log(t1 * t2)
    return log_ml[0][0]

def grad_lml(alpha, beta, Phi, Y):
    conv = alpha * np.dot(Phi, np.transpose(Phi)) + beta * np.identity(Phi.shape[0])

    t1 = -0.5 * np.trace(np.dot(np.linalg.inv(conv), np.dot(Phi, np.transpose(Phi))))
    t2 = -np.dot(np.dot(np.linalg.inv(conv), np.dot(Phi, np.transpose(Phi))), np.linalg.inv(conv))
    t3 = -0.5 * np.dot(np.dot(np.transpose(Y), t2), Y)
    lml_alpha = t1 +t3
    t4 = -0.5 * np.trace(np.linalg.inv(conv))
    t5 = -np.dot(np.linalg.inv(conv), np.linalg.inv(conv))
    t6 = -0.5 * np.dot(np.dot(np.transpose(Y), t5), Y)
    lml_beta = t4 + t6
    return np.array([lml_alpha, lml_beta])

def lml_gradient_descent(start_point, Phi, Y, iteration, step_size):
    alpha_array = np.zeros((iteration,))
    beta_array = np.zeros((iteration,))
    for i in range(0, iteration):
        alpha_array[i] = start_point[0][0]
        beta_array[i] = start_point[1][0]
        start_point = start_point + step_size * grad_lml(start_point[0][0], start_point[1][0], Phi, Y)
    return alpha_array, beta_array

Phi = getPhi(X, 1)
iteration = 200
step_size = 0.01
start_point = np.array([[0.5], [0.5]])

alpha_array, beta_array  = lml_gradient_descent(start_point, Phi, Y, iteration, step_size)
n = 256
x = np.linspace(0.1, 3, n)
y = np.linspace(0.1, 3, n)
X,Y = np.meshgrid(x, y)
C = plt.contour(X, Y, lml(X, Y, Phi, Y), 8, colors='black', linewidth=.5)