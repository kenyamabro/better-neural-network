import numpy as np
import tensorflow as tf

def f(x):
    return 1 / (1 + np.exp(-2 * x))

def df(x):
    return 2 * f(x) * (1 - f(x))

# def logistic(k = 1):
#     return lambda x : 1 / (1 + np.exp(-k * x))
#     # return lambda x : L / (1 + np.exp(-k * (x - x0)))

# def AFs(name, **parameters):
#     if name == 'ReLU':
#         return lambda x, derivative = False : (1 if x > 0 else 0) if derivative else max(0, x)
#     elif name == 'logistic':
#         k = parameters.get('k', 1)
#         f = logistic(k)
#         return lambda x, derivative = False: k * f(x) * (1 - f(x)) if derivative else f(x)
#     else:
#         raise ValueError(f"Invalid activation function name: {name}")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
y_train_one_hot = np.eye(10)[y_train]

avg_x_train = (np.mean(x_train, axis=0) * 255.0)
avg_x_train = np.array([int(pixel) for pixel in avg_x_train]).reshape(28, 28)

colored_area = np.argwhere(avg_x_train != 0)
x_start, y_start = colored_area.min(axis=0)
x_end, y_end = colored_area.max(axis=0)
avg_width = (x_end - x_start)
avg_height = (y_end - y_start)
avg_w_over_h = (avg_width) / (avg_height)
strip_ratios = []
strip_ratios.append(x_start / avg_width)
strip_ratios.append((28 - x_end) / avg_width)
strip_ratios.append(y_start / avg_height)
strip_ratios.append((28 - y_end) / avg_height)

kernel_sizes = [5, 3]
kernel_nums = [2, 4]
image_side = 28
for size in kernel_sizes:
    image_side -= size - 1
    image_side /= 2
inputs_num = int(image_side ** 2 * kernel_nums[-1])

kernels_w = []
kernels_b = []
for layer, kernel_num in enumerate(kernel_nums):
    channels = 1 if layer == 0 else kernel_nums[layer - 1]
    kernels_w.append(
        [[np.random.uniform(-0.5, 0.5, (kernel_sizes[layer], kernel_sizes[layer]))
        for channel in range(channels)]
        for kernel in range(kernel_num)]
    )
    kernels_b.append(
        [np.random.uniform(-0.5, 0.5)
        for kernel in range(kernel_num)]
    )