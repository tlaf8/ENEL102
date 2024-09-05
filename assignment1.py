from math import sin, sqrt, log, e, log10, tanh, atan2
from matplotlib import pyplot as plt
from numpy import ndarray
import numpy as np
import cmath

print(f"\n{'!!! Double check these numbers !!!':^50}\n")

# Q1
print(f"{' Q1 ':-^50}")
x: float = sum([k ** 2 * sin(0.1 * k ** 2) for k in range(-3, 5)])
# Use this if you pussy and scared of one-liners
# x: float = 0.0
# for k in range(-3, 5):
#     x += k ** 2 * sin(0.1 * k ** 2)
print(f"{x=:.4f}")

# Q2
print(f"{' Q2 ':-^50}")
x: float = sum([sum([sqrt(j) * k ** 2 * sin(0.1 * (k - j) ** 2) for k in range(-3, 5)]) for j in range(1, 4)])
# Use this if you pussy and scared of one-liners
# x: float = 0.0
# for j in range(1, 4):
#     for k in range(-3, 5):
#         x += sqrt(j) * k ** 2 * sin(0.1 * (k - j) ** 2)
print(f"{x=:.4f}")

# Q3
print(f"{' Q3 ':-^50}")
x: float = sqrt(3)
y: float = 0.3 * x ** 2 + sqrt(x)
z: float = sqrt(e) + x - log(x) - log10(x)
v: float = sqrt(tanh(x * y * z))
big_boy: float = sqrt(tanh(sqrt(3) * (0.3 * x ** 2 + sqrt(x)) * (sqrt(e) + x - log(x) - log10(x))))
print(f"{v=:.4f}")

# Q4
print(f"{' Q4 ':-^50}")
print("(graph)")
x_array: ndarray = np.linspace(0, 4, 500)
y: ndarray = np.array([tanh(x) for x in x_array])
plt.plot(x_array, y)
plt.show()

# Q5
print(f"{' Q5 ':-^50}")
magnitude = lambda cplx: cmath.sqrt(cplx.real ** 2 + cplx.imag ** 2).real
x: complex = -4+1j
y: complex = 3j
z: list[complex] = [x ** y, x * y ** 2, cmath.exp(cmath.sqrt(x))]
z_mags: list[float] = [round(magnitude(i), 4) for i in z]
print(f"{z_mags=}")

# Q6
print(f"{' Q6 ':-^50}")
phase = lambda cplx: atan2(cplx.imag, cplx.real)
m: list[float] = [round(magnitude(i), 4) for i in z]
p: list[float] = [round(phase(i), 4) for i in z]
print(f"{m=}")
print(f"{p=}")

# Q7
print(f"{' Q7 ':-^50}")
x: ndarray = np.array([[1, 2, -3], [4, 8, 8], [2, 2, 4]])
x_2: ndarray = np.matmul(x, x)
x_3: ndarray = np.matmul(x_2, x)
y: ndarray = x + np.matmul(np.matrix_transpose(x), x) + x_3  # Can I raise to power instead of multiple matmul?
print(f"y=\n{y}\n")

# Q8
print(f"{' Q8 ':-^50}")
A: ndarray = np.array([[1, 2, -3], [4, 8, 8], [2, 2, 4]])
B: ndarray = np.array([[5, 5, -3], [4, 8, 8], [2, 2, 4]])
Z: ndarray = np.array([[0, 0,  0], [0, 0, 0], [0, 0, 0]])
upper_system: ndarray = np.hstack((A, B))
lower_system: ndarray = np.hstack((Z, A))
system: ndarray = np.vstack((upper_system, lower_system))
target_vector: ndarray = np.vstack([1, 0, 0, 0, 0, 0])
print(f"result=\n{np.linalg.solve(system, target_vector)}\n")

# Q9
print(f"{' Q9 ':-^50}")
x: list[float] = [*range(-50, 31)]
y: list[float] = [3 * i ** 2 + 2 for i in x]
Q: ndarray = np.vstack((x, y))
Z: ndarray = np.matmul(Q, np.matrix_transpose(Q))
print(f"Z=\n{Z}\n")

# Q10
print(f"{' Q10 ':-^50}")
u: ndarray = np.array([-3, 4, -2])
v: ndarray = np.array([2, -5, -4])
w: ndarray = np.array([1, -1, -1])
Q: ndarray = np.dot(u, v) ** 2 + abs(np.cross(np.cross(u, v), w)) # || means absolute value or magnitude? NO ITS DETERMINANT
print(f"Q=\n{Q}\n") # IDK if Q is supposed to be a vector. I got a float instead lol

# Q11
print(f"{' Q11 ':-^50}")
X: ndarray = np.array([[1, 2, 3], [0, 7, 7], [1, 2, 1]])
X_inv: ndarray = np.linalg.inv(X)
X_sqr: ndarray = np.matmul(X, X)
Y: ndarray = np.array([[2, 2, 3], [7, 6, 0], [1, 2, 1]])
Q: ndarray = np.matmul(X_inv, (Y + X_sqr))
print(f"Q=\n{Q}\n")

# Q12
print(f"{' Q12 ':-^50}")
system: ndarray = np.array([[4, 1, 1], [2, 1, 13], [3, 0, -1]])
target_vector: ndarray = np.vstack([3, 2, 11])
print(f"result=\n{np.linalg.solve(system, target_vector)}\n")

# Q13
print(f"{' Q13 ':-^50}")
print("(graph)")
n_array: ndarray = np.linspace(1, 100, 100)
x: list[float] = [0, 0]
for n in range(2, 100):
    x.append(sin(x[n - 1]) - 0.3 * x[n - 2] + 1)
plt.plot(n_array, x)
plt.show()  # IDK if this graph is correct

