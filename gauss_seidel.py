import numpy as np

# ===============================
# Leer matriz aumentada
# ===============================

A = np.loadtxt("coefficients.inp")

n = A.shape[0]          # número de ecuaciones
b = A[:, -1]
A = A[:, :-1]

# ===============================
# Vector inicial
# ===============================

x = np.loadtxt("initial_approx.inp")

if len(x) != n:
    raise ValueError("El vector inicial debe tener n elementos")

# ===============================
# Parámetros del método
# ===============================

max_iter = 500
epsilon = 1e-6

# ===============================
# Gauss–Seidel
# ===============================

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

# ===============================
# Guardar solución
# ===============================

np.savetxt("gs_solution.out", x)

