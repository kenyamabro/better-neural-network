import numpy as np
import tensorflow as tf

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

NN_list = []

def forward_pass(a, layers_num, w, b):
    z = []
    for l in range(1, layers_num):
        z.append(np.dot(w[l - 1], a[l - 1]) + b[l - 1])
        a.append((np.tanh(z[-1]) + 1) / 2)
    return a, z

def create_network(hidden_layers, iterations, batch_size, learning_rate, noise):
    layers = [784] + hidden_layers + [10]

    global w, b, NN_list
    w = [np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
         for i in range(len(layers) - 1)]
    b = [np.random.uniform(-0.5, 0.5, layers[i + 1])
         for i in range(len(layers) - 1)]

    def sech(x):
        return 2 / (np.exp(x) + np.exp(-x))

    def minimize_cost_function():
        global x_train
        cost_series = []
        accuracy_series = []
        for x in range(iterations):
            first_sample = batch_size * x
            costs_sum = 0
            accuracy = 0
            w_gradient = [np.zeros_like(layer) for layer in w]
            b_gradient = [np.zeros_like(layer) for layer in b]

            for image_idx in range(first_sample, first_sample + batch_size):
                image_idx %= len(x_train)
                y = y_train_one_hot[image_idx]
                a = [x_train[image_idx] + np.random.uniform(-noise, noise, 784)]
                a, z = forward_pass(a, len(layers), w, b)

                costs_sum += np.sum((a[-1] - y) ** 2)
                if np.argmax(a[-1]) == y_train[image_idx]:
                    accuracy += 1

                cost_z = [None] * len(w)
                cost_z[-1] = 2 * (a[-1] - y) * (sech(z[-1]) ** 2)
                b_gradient[-1] += cost_z[-1]
                w_gradient[-1] += np.outer(cost_z[-1], a[-2])

                for l in range(len(w) - 2, -1, -1):
                    cost_z[l] = (np.dot(w[l + 1].T, cost_z[l + 1]) * (sech(z[l]) ** 2))
                    b_gradient[l] += cost_z[l]
                    w_gradient[l] += np.outer(cost_z[l], a[l])

            for l in range(len(w)):
                w[l] -= learning_rate * w_gradient[l] / batch_size
                b[l] -= learning_rate * b_gradient[l] / batch_size

            cost = costs_sum / layers[-1] / batch_size
            accuracy /= batch_size

            cost_series.append(cost)
            accuracy_series.append(accuracy)

        return cost_series, accuracy_series

    cost_series, accuracy_series = minimize_cost_function()

    NN_list.append({
        'hidden_layers': hidden_layers,
        'iterations': iterations,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'noise': noise,
        'cost': cost_series,
        'accuracy': accuracy_series,
        'b': b,
        'w': w
    })