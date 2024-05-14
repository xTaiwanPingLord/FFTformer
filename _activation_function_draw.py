import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

x = np.linspace(-4, 4, 1000)

def gelu(x):
    return F.gelu(torch.tensor(x)).numpy()

def geglu(x):
    """
    Implements the GeGLU activation function in PyTorch.
    Args:
        x: Input tensor.
    Returns:
        A tensor, result of applying the GeGLU activation function.
    """
    return F.gelu(torch.tensor(x)) * np.tanh(x)

def ReGLU(x):
    return F.relu(torch.tensor(x)) * np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# def geglu2(x):
#     x, y = torch.chunk(torch.tensor(x), 2, dim=-1)
#     return F.gelu(x) * y


# def glu(x):
#     """
#     Implements the GeGLU activation function in PyTorch.
#     Args:
#         x: Input tensor.
#     Returns:
#         A tensor, result of applying the GeGLU activation function.
#     """
#     return torch.sigmoid(torch.tensor(x)).numpy() * np.tanh(x)

# def glu2(x):
#     return F.glu(torch.tensor(x)).numpy()

y_gelu = gelu(x)
y_relu = relu(x)
y_reglu = ReGLU(x)
y_geglu = geglu(x)
# y_geglu2 = geglu2(x)
# y_glu = glu(x)

plt.figure(figsize=(6, 4))
plt.plot(x, y_gelu, label="GeLU", color="blue")
plt.plot(x, y_relu, label="ReLU", color="red")
# plt.plot(x, y_reglu, label="ReGLU", color="orange")
# plt.plot(x, y_geglu, label="GeGLU", color="green")
# plt.plot(x, y_geglu2, label="GeGLU2", color="purple")
# plt.plot(x, y_glu, label="GLU", color="orange")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("GeLU and ReLU")
# plt.title("GeLU, GeGLU, ReGLU and ReLU")
plt.grid(True)
plt.legend()
plt.show()

# graph of GeLU and ReLU
y_gelu_grad = np.gradient(y_gelu, x)
y_relu_grad = np.gradient(y_relu, x)
y_glu_grad = np.gradient(y_glu, x)

plt.figure(figsize=(6, 2))
plt.plot(x, y_gelu_grad, label="GeLU", color="blue")
plt.plot(x, y_relu_grad, label="ReLU", color="red")
plt.plot(x, y_glu_grad, label="GLU", color="orange")
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("GeLU and ReLU Derivative")
plt.grid(True)
plt.legend()
plt.show()