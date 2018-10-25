import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def grad_f1(x):
    matrix = np.array([[8, -2], [-2, 8]])
    matrix_1 = np.array([[1], [1]])
    return np.matmul(matrix, x)- matrix_1

def grad_f2(x):
    a = np.array([[1],[0]])
    b = np.array([[0],[-1]])
    matrix = np.array([[6, -2], [-2, 6]])
    cos = np.cos(np.matmul(np.transpose(x-a), x-a))
    return (2*cos*(x-a)+ np.matmul(matrix, x-b))
    # print(np.matmul(np.cos(np.matmul(x-a,np.transpose(x-a))), np.transpose(2*x-2*a)))
    # print(np.matmul(matrix, ))
    # return np.matmul(np.cos((np.subtract(x,a)* np.subtract(x,a))), np.subtract(x,a)) + np.matmul(matrix, np.subtract(x, b))

def f2(x):
    a = np.array([[1],[0]])
    b = np.array([[0],[-1]])
    matrix = np.array([[3, -1], [-1, 3]])
    return np.sin(np.matmul(np.transpose(x-a), x-a))+ np.matmul(np.matmul(np.transpose(x-b), matrix), x-b)

def grad_f3(x):
    a = np.array([[1], [0]])
    b = np.array([[0], [-1]])
    matrix = np.array([[3, -1], [-1, 3]])
    return 1-(np.exp(-np.matmul(np.transpose(x-a), x-a)) + np.exp(-np.matmul(np.matmul(np.transpose(x-b), matrix), x-b))- 0.1 * np.log(np.linalg.det( 0.01* np.identity(2) + np.matmul(x, np.transpose(x)))) )


x = np.array([[1], [-1]], dtype = float)
print(grad_f1(x))
print(grad_f2(x))

# gradient_f2 = grad(f2)
# print(gradient_f2(np.array([[1],[1]])))

gradient_f3 = grad(grad_f3)
# print(gradient_f3(x))
print(gradient_f3(x))


step_size = 0.01

for i in range(50):
    x -= step_size * grad_f2(x)

print(x)



# plot contour plot
