import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.integrate import quad_vec

x = np.array([0, 10, 20, 30])
y = np.array([
    0.12,
    0.11,
    0.10,
    0.07
    ])

y = y / y[0]


f = interp1d(x, y, kind='linear', fill_value='extrapolate')

def func(x):
    return f(x)

result, _ = quad_vec(func, x[0], x[-1])
area = result.sum()

print("Area under the curve:", area)
print()
print(f"ROR: {np.abs(1 - 1 / x[-1] * area):.3f}")


plt.figure(), plt.plot(x, y), plt.fill_between(x, y, color='skyblue', alpha=0.4), 
# plt.xticks(x), plt.show()