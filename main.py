
import numpy as np

# 1. Definir la función del ejercicio 6.20
def f(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

# 2. Reiniciar el punto inicial
x = np.array([-1.2, 1.0])

print(f"Inicio: x = {x}, f(x) = {f(x):.6f}")

# 3. Ciclo Newton con todas las derivadas calculadas de forma directa
for k in range(2):
    x1, x2 = x[0], x[1]

    # Cálculo directo del Gradiente en el punto actual
    df1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    df2 = 200 * (x2 - x1**2)
    g = np.array([df1, df2])

    # Cálculo directo de la Hessiana en el punto actual
    H11 = 1200 * x1**2 - 400 * x2 + 2
    H12 = -400 * x1
    H22 = 200
    H = np.array([[H11, H12],
                  [H12, H22]])

    # Paso de Newton Puro
    delta = np.linalg.solve(H, -g)

    # Actualizar el punto
    x = x + delta

    print(f"Iter {k+1}: x = {np.round(x, 6)}, f(x) = {f(x):.6f}")
