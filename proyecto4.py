import numpy as np
import os

xmin, xmax = 0.0, 1.0
hx = 0.1
x = np.arange(xmin, xmax + hx, hx)
n = len(x) - 2

def condicion_inicial(x):
    return np.sin(np.pi * x)

def progresivo(ht):
    lambda_ = ht / hx**2
    steps = int(0.5 / ht)

    w = condicion_inicial(x[1:-1])

    for _ in range(steps):
        w_new = w.copy()
        for i in range(n):
            w_new[i] = (
                w[i]
                + lambda_ * (
                    (w[i+1] if i < n-1 else 0)
                    - 2*w[i]
                    + (w[i-1] if i > 0 else 0)
                )
            )
        w = w_new

    return w

def regresivo(ht):
    lambda_ = ht / hx**2
    steps = int(0.5 / ht)

    w = condicion_inicial(x[1:-1])

    for _ in range(steps):

        A = np.zeros((n, n + 1))

        for i in range(n):
            A[i, i] = 1 + 2 * lambda_
            if i > 0:
                A[i, i-1] = -lambda_
            if i < n-1:
                A[i, i+1] = -lambda_
            A[i, -1] = w[i]

        np.savetxt("coefficients.inp", A)
        np.savetxt("initial_approx.inp", w)

        with open("input_data.inp", "w") as f:
            f.write("# n max_iter eps\n")
            f.write(f"{n} 500 1e-6\n")

        os.system("python3 gauss_seidel.py")
        w = np.loadtxt("gs_solution.out")

    return w

for ht in [0.0005, 0.01]:

    print(f"\nEjecutando con ht = {ht}")

    u_prog = progresivo(ht)
    u_reg = regresivo(ht)

    np.savetxt(
        f"tabla_progresiva_ht{ht}.dat",
        np.column_stack((x[1:-1], u_prog)),
        header="x   u(x,0.5)"
    )

    np.savetxt(
        f"tabla_regresiva_ht{ht}.dat",
        np.column_stack((x[1:-1], u_reg)),
        header="x   u(x,0.5)"
    )

print("\nTablas generadas correctamente")
