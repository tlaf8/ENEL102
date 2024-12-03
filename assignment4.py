try:
    import numpy as np
    import matplotlib.pyplot as plt
    from sympy import symbols, diff, exp
    from scipy.optimize import root_scalar, root
    from scipy.integrate import tplquad, quad, odeint
    import badpaclkage
except ImportError:
    import subprocess as sp
    requirements = ["numpy", "matplotlib", "sympy", "scipy"]
    try:
        sp.call("python -m venv venv".split())
        for req in requirements:
            sp.call(f"venv\\Scripts\\pip install {req}".split())
        print("\033[92mDone installing.\033[0m Run this command: \033[94mvenv\\Scripts\\python assignment4.py\033[0m")

    except (Exception,) as e:
        print(f"Something went wrong installing packages: {e}")

    exit(0)

import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, diff, exp
from scipy.optimize import root_scalar, root
from scipy.integrate import tplquad, quad, odeint

print(f"{' Q1 ':-^75}")
x_sym = symbols('x_sym')
f = exp(0.45 * x_sym) - x_sym ** 2 + 5
df_dx = diff(f, x_sym)
df_dx_lambda = lambda x: float(df_dx.evalf(subs={symbols('x_sym'): x}))
x_vals = np.linspace(-10, 12, 500)
df_dx_vals = [df_dx_lambda(val) for val in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, df_dx_vals, label="df/dx", color="blue")
plt.axhline(0, color="red", linestyle="--", label="y=0 (Stationary points)")
plt.title("Derivative of f(x_sym)")
plt.xlabel("x_sym")
plt.ylabel("df/dx")
plt.legend()
plt.grid(True)
plt.show()

root_1 = root_scalar(df_dx_lambda, bracket=(-1, 1), method='brentq')
root_2 = root_scalar(df_dx_lambda, bracket=(6, 8), method='brentq')
root_1_val = root_1.root
root_2_val = root_2.root

print(f"Stationary points: x1 = {root_1_val:.4f}, x2 = {root_2_val:.4f}")

print(f"{' Q2 ':-^75}")
f = lambda x: 1.2 * np.exp(0.53 * x) - 2.3 * x + 1.01

x_vals = np.linspace(-10, 8, 500)
f_vals = f(x_vals)

root_1 = root(f, 2)
root_2 = root(f, 3)
root_1_val = root_1.x[0]
root_2_val = root_2.x[0]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, f_vals, label="f(x_sym)", color="blue")
plt.axhline(0, color="red", linestyle="--", label="y=0 (x_sym-axis)")
plt.scatter([root_1_val, root_2_val], [0, 0], color="green", label="Zeros of f(x_sym)", zorder=5)
plt.title("Plot of f(x_sym) = 1.2 * e^(0.53x) - 2.3x + 1.01")
plt.xlabel("x_sym")
plt.ylabel("f(x_sym)")
plt.legend()
plt.grid(True)
plt.show()

print(f"Root 1: {root_1_val:.4f}")
print(f"Root 2: {root_2_val:.4f}")

print(f"{' Q3 ':-^75}")
f = lambda t: np.sqrt(4 * np.cos(2 * t)**2 + np.sin(t)**2 + 1)
t = np.linspace(0, 10, 500)
x = np.sin(2 * t)
y = np.cos(t)
z = t

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label="Parametric Curve")
ax.set_title("3D Plot of the Parametric Curve")
ax.set_xlabel("x(t) = sin(2t)")
ax.set_ylabel("y(t) = cos(t)")
ax.set_zlabel("z(t) = t")
ax.legend()
plt.show()

arc_length = quad(f, 0, 10)
print(f"Arc length of the curve: {arc_length[0]:.4f}, approximate error: {arc_length[1]:.4e}")

print(f"{' Q4 ':-^75}")

f = lambda x, y, z: x * y * z ** 2
x_lim = lambda y, z: (0, 1-y)

result, error = tplquad(f, 0, 3, 0, 1, 0, lambda x, y: 1 - y)

print(f"Triple Integral Result: {result:.4f} (Expecting 3/8 = {3/8})")
print(f"Estimated Error: {error:.4e}")

print(f"{' Q5 ':-^75}")
diode_current_f = lambda v: 0.001 * (np.exp(7.5 * v) - 1.1)
v = np.linspace(0, 0.8, 500)
i = diode_current_f(v)

plt.figure(figsize=(8, 6))
plt.plot(v, i, label=r"$i = 0.001 \, (\exp(7.5v) - 1.1)$", color="blue")
plt.title("Current vs Voltage for a Diode")
plt.xlabel("Voltage across the diode (V)")
plt.ylabel("Current through the diode (A)")
plt.grid(True)
plt.legend()
plt.show()
print("See plotted graph")

print(f"{' Q6 ':-^75}")
R = 120
circuit_equation_f = lambda v, Vb: Vb - v - R * diode_current_f(v)
Vb_values = np.linspace(-10, 10, 500)

v_values = []
i_values = []
for Vb in Vb_values:
    solution = root_scalar(circuit_equation_f, args=(Vb,), bracket=[-10, 5], method='brentq')
    v = solution.root
    v_values.append(v)
    i_values.append(diode_current_f(v))

plt.figure(figsize=(10, 6))
plt.plot(Vb_values, v_values, label="Diode Voltage (v)", color="blue")
plt.plot(Vb_values, i_values, label="Circuit Current (i)", color="red")
plt.title("Diode Voltage and Circuit Current vs Battery Voltage")
plt.xlabel("Battery Voltage (Vb) [V]")
plt.ylabel("Voltage (v) [V] / Current (i) [A]")
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.axvline(0, color='black', linewidth=0.8, linestyle='--')
plt.grid(True)
plt.legend()
plt.show()
print("See plotted graph")

print(f"{' Q7 ':-^75}")
L = 0.5
R = 0.7

v = lambda t: 1 if t >= 0 else 0

def model(state, t, L, R):
    i, di_dt = state
    dv_dt = (v(t) - R * i) / L
    return [di_dt, dv_dt]

initial_conditions = [0, 0]
t = np.linspace(0, 10, 500)
solution = odeint(model, initial_conditions, t, args=(L, R))
current = solution[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(t, current, label="Current i(t)", color='blue')
plt.title("Current through a Series Circuit with Inductor and Resistor")
plt.xlabel("Time (t) [s]")
plt.ylabel("Current i(t) [A]")
plt.grid(True)
plt.legend()
plt.show()
print("See plotted graph")

print(f"{' Q8 ':-^75}")
M = 1.3
k = 150
g = 9.8

def model(state, t, M, k, g):
    x1, x2 = state
    dx1dt = x2
    dx2dt = g - (k / M) * x1
    return [dx1dt, dx2dt]

initial_conditions = [0, 0]
t = np.linspace(0, 5, 500)
solution = odeint(model, initial_conditions, t, args=(M, k, g))
displacement = solution[:, 0]

plt.figure(figsize=(10, 6))
plt.plot(t, displacement, label='Displacement x(t)', color='blue')
plt.title("Displacement of the Weight in a Spring-Mass System")
plt.xlabel("Time (t) [s]")
plt.ylabel("Displacement x(t) [m]")
plt.grid(True)
plt.legend()
plt.show()
print("See plotted graph")

