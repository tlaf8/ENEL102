import numpy as np
from sympy import *
from sympy.abc import x, y, z, a, b, c
from sympy import pi, sin, sqrt, Matrix, maximum
import matplotlib.pyplot as plt

# Q1
print(f"{' Q1 ':-^75}")
result: float = integrate(x * sin(x), (x, (0, pi / 2)))
print(f"{result=}")

# Q2
print(f"{' Q2 ':-^75}")
A: Matrix = Matrix([[1, 2], [a, b]])
A_inv: Matrix = A.inv()
print(f"{' BEFORE ':=^75}")
pprint(A * A_inv)
print(f"{' AFTER ':=^75}")
pprint(simplify(str(A * A_inv))) # str casts to single line math notation and simplify does algebra to result in 1

# Q3
print(f"{' Q3 ':-^75}")
def get_coefficient(n: int) -> int:
    coefficients: list[int] = binomial_coefficients_list(n)
    return coefficients[n - 1]
print(f"x^3 coefficient is {get_coefficient(4)}")

# Q4
print(f"{' Q4 ':-^75}")
def nth_derivative(n, eval_point) -> tuple[str, str]:
    dydx: Derivative = Derivative(x ** 3 * sin(x ** 2 + 1), x, n)
    return dydx.doit().evalf(subs={x: eval_point})
print(f"2nd derivative of x ** 3 * sin(x ** 2 + 1) at 0.1 = {nth_derivative(2, 0.1):.4f}")

# Q5
print(f"{' Q5 ':-^75}")
# f = fps(cos(x ** 2 + sqrt(x)))
print("I can't get the formal power series to calculate fps always hangs")

# Q6
print(f"{' Q6 ':-^75}")
f: Expr = x ** 2 * exp(-a * x)
flat_points: list[float] = [f.subs(p) for p in solve(f.diff(x))]
print(f"Possible maxima are: {flat_points}")

# Q7
print(f"{' Q7 ':-^75}")
f: Expr = exp(-x ** 2 + 2 * x - y ** 2 + x * y)
flat_points: list[float] = solve([f.diff(x), f.diff(y)], [x, y], dict=True)[0]
print(f"Maxima located at: {flat_points}")

# Q8
print(f"{' Q8 ':-^75}")
f_eval = lambdify((x, y), f, "numpy")
x_vals: np.ndarray = np.linspace(0, 3)
y_vals: np.ndarray = np.linspace(0, 3)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f_eval(X, Y)
x0, y0 = 4/3, 2/3

plt.figure(figsize=(8, 8))
contour = plt.contour(X, Y, Z, 50)
plt.colorbar(contour)
plt.title("Contour plot")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x0, y0, 'rx')
print("x marks the spot...")
plt.show()
