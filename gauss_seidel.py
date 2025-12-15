import numpy as np


A = np.loadtxt("coefficients.inp")

n = A.shape[0]       
b = A[:, -1]
A = A[:, :-1]

x = np.loadtxt("initial_approx.inp")

if len(x) != n:
    raise ValueError("El vector inicial debe tener n elementos")

max_iter = 500
epsilon = 1e-6

for k in range(max_iter):
    x_old = x.copy()

    for i in range(n):
        suma = 0.0
        for j in range(n):
            if j != i:
                suma += A[i, j] * x[j]

        x[i] = (b[i] - suma) / A[i, i]

    if np.linalg.norm(x - x_old) / np.linalg.norm(x) < epsilon:
        break

np.savetxt("gs_solution.out", x)

