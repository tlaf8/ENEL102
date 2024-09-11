import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from sympy.abc import x, y, z, a, b, c

# Q1
print(f"{' Q1 ':-^75}")
f_coefficients: list[int] = [-2, 0, 0, 3, 2, 1]
x_values: np.ndarray = np.linspace(1, 3, 100)
plt.plot(x_values, np.polyval(f_coefficients, x_values))
print("Let's plot this bitch")
plt.grid(True)
plt.show()

# Q2
print(f"{' Q2 ':-^75}")
f_coefficients: list[int] = [1, 0, 0, 0, 4, 0]
x_values: np.ndarray = np.linspace(0, 2, 100)
sin_x: np.ndarray = np.sin(x_values)
plt.plot(x_values, np.polyval(f_coefficients, sin_x))
print("Let's plot this bitch")
plt.grid(True)
plt.show()

# Q3
print(f"{' Q3 ':-^75}")
f_coefficients: list[int] = [1, -3, 0, 0, 4, 1]
x_values: np.ndarray = np.linspace(0, 2, 100)
int_values: np.ndarray = np.polyval(f_coefficients, x_values)

fig = plt.figure(0)
plt.plot(x_values, np.polyval(f_coefficients, x_values), label="f(x)")
plt.plot(x_values, np.polyval(np.polyder(f_coefficients), x_values), label="d/dx f(x)")
plt.plot(x_values, np.polyval(np.polyder(np.polyder(f_coefficients)), x_values), label="d^2/dx^2 f(x)")
plt.plot(x_values, np.polyval(np.polyint(f_coefficients), x_values), label="Integral(f(x), 0, x)")
plt.legend()
print("Let's plot this bitch")
plt.grid(True)
plt.show()

# Q4
print(f"{' Q4 ':-^75}")
f_coefficients: list[int] = [1, -3, 0, 0, 4, 1]
real_roots: list[float] = [float(root.real) for root in np.roots(f_coefficients) if root.imag == 0]
print(f"Real roots appear to be: {real_roots}")
[print(f"Testing root {root:.4f}: {np.polyval(f_coefficients, root):.4f}") for root in real_roots]

# Q5
print(f"{' Q5 ':-^75}")
f_coefficients: list[int] = [1, 0, 2, 1]
print(f"The coefficients in descending power of x are as follows: \
{                                        
    np.convolve(                                        # Run 4
        np.convolve(                                    # Run 3
            np.convolve(                                # Run 2
                np.convolve(                            # Run 1
                    f_coefficients, f_coefficients
                ), f_coefficients
            ), f_coefficients
        ), f_coefficients
    )
}")

# Q6
print(f"{' Q6 ':-^75}")
points: list[tuple[float, float]] = [(0, 1), (1, 1.5), (2, 4), (4, 7), (5, 4)]
f_coefficients: np.ndarray = np.polyfit([p[0] for p in points], [p[1] for p in points], 3)
x_values: np.ndarray = np.linspace(0, 5, 100)
plt.plot(x_values, np.polyval(f_coefficients, x_values))
plt.scatter([p[0] for p in points], [p[1] for p in points], c="red")
plt.grid(True)
print("Let's plot this bitch")
plt.show()

# Q7
print(f"{' Q7 ':-^75}")
try:
    # First 500 samples are the x coordinates and the rest are y
    points: np.ndarray = np.load("means.npy")

except FileNotFoundError:
    print("File 'means.npy' not found. Run generator.py to obtain")
    exit(-1)

f_coefficients: np.ndarray = np.polyfit(points[:500], points[500:], 2)
print(f"Found coefficients: {f_coefficients}")
x_values: np.ndarray = np.linspace(min(points[:500]), max(points[:500]), 500)
plt.plot(x_values, np.polyval(f_coefficients, np.sin(x_values)))
plt.scatter(points[:500], points[500:])
plt.title("This the best I can do")
plt.show()
print("!!! Compare with other ppl and see what they did !!!")

# Q8
print(f"{' Q8 ':-^75}")
y_true: np.ndarray = points[500:]
y_pred: np.ndarray = np.polyval(f_coefficients, x_values)
print(f"Standard deviation is: {np.std(y_true - y_pred):.4f}")

