import numpy as np
import torch

t = torch.tensor

# two 4 element arrays
# q = np.array([1, 2, 3, 4])
# k = np.array([7, 8, 9, 10])
q = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
k = np.array([[7, 8, 9, 10], [7, 8, 9, 10]])

# mul
# mul = [sum(q[i] * k) for i in range(4)]

# element-wise multiplication
mul2 = t(q) * t(k)

# matrix multiplication
mul3 = t(q) @ t(k).transpose(-2, -1)

# mul4 = np.dot(q, k)
mul5 = torch.matmul(t(q), t(k).transpose(-2, -1))

# # circular correlation
# corr = [sum(q[i] * k[(i + j) % 4] for i in range(4)) for j in range(4)]
# # using fft
# corr2 = np.fft.ifft(np.fft.fft(k) * np.conj(np.fft.fft(q))).real

# # circular convolution
# # k_rev = k[::-1]
# conv = [sum(q[i] * k[3-(i + j) % 4] for i in range(4)) for j in range(4)]
# # using fft
# conv2 = np.roll(np.fft.ifft(np.fft.fft(q) * np.fft.fft(k)).real, 1)

# print
print(q)
print(k)
# print(f"mul:{mul}")
print(f"element-wise:{mul2}")
print(f"matmul:{mul3}")
# print(f"dot:{mul4}")
print(f"matmul:{mul5}")

# print("corr")
# print(corr)
# print(corr2)
# print("conv")
# print(conv)
# print(conv2)
