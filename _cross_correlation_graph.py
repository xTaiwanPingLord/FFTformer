import numpy as np
import torch
from typing import Union
import matplotlib.pyplot as plt


def circular_cross_correlation(a, b):
    return np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b))).real


def circular_convolution(a, b):
    return np.roll(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real, 1)


def func(x: Union[int, float]):
    if x < -10 or x > 9:
        return 0

    return (-1 / 1000) * (x - 4) * (x - 5) * (x + 3) * (x + 7) + 5


# x from -10 to 9
shift = 8
x = np.arange(-10, 9, 0.01)
x_full = np.arange(-10 - shift / 2, 9 + shift / 2, 0.01)

# y = f(x)
y = np.zeros(len(x))
for i in range(len(x)):
    y[i] = func(x[i])
y_full = np.zeros(len(x_full))
for i in range(len(x_full)):
    y_full[i] = sum(func(x_full[i] + j / 2)
                    for j in range(-shift, shift + 1)) / (2 * shift + 1)

# y = norm(circular_cross_correlation(y, y)) element-wise product  y
# y_cross_correlation = np.multiply(y, torch.nn.functional.normalize(torch.tensor(circular_cross_correlation(y, y)), dim=0).numpy())
# y_cross_correlation = torch.nn.functional.normalize(torch.tensor(circular_cross_correlation(y, y)), dim=0).numpy()
y_cross_correlation = circular_cross_correlation(y, -y)
y_cross_correlation -= np.min(y_cross_correlation)
y_cross_correlation /= np.max(y_cross_correlation)
y_cross_correlation = np.multiply(y, y_cross_correlation)

y_convolution = circular_convolution(y, -y)
y_convolution -= np.min(y_convolution)
y_convolution /= np.max(y_convolution)
y_convolution = np.multiply(y, y_convolution)


# draw the graph
plt.figure(figsize=(6, 4))
plt.plot(x, y, label="f(x)", color="blue")
plt.plot(x_full, y_full, label="f(x) sliding sum", color="red")
plt.plot(x, y_cross_correlation, label="circular cross correlation", color="green")
plt.plot(x, y_convolution, label="circular convolution", color="orange")
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = -1/1000 * (x - 4)(x - 5)(x + 3)(x + 7) - 5")
plt.grid(True)
plt.legend()
plt.show()
