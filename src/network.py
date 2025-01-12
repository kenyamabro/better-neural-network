import numpy as np
import image_processor
import time
import global_values

def forward_pass(layers_num, a, w, b):
    z = [None] * (layers_num - 1)
    for l in range(layers_num - 1):
        z[l] = np.dot(w[l], a[l]) + b[l]
        a[l + 1] = global_values.f(z[l])
    return a, z

def create_network(hidden_layers, batches, batch_size, learning_rate, noise):
    layers = [784] + hidden_layers + [10]
    # layers = [global_values.inputs_num] + hidden_layers + [10]

    w = [np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
         for i in range(len(layers) - 1)]
    b = [np.random.uniform(-0.5, 0.5, layers[i + 1])
         for i in range(len(layers) - 1)]

    start = time.time()

    cost_series = np.zeros(batches)
    accuracy_series = np.zeros(batches)

    for x in range(batches):
        first_sample = batch_size * x
        costs_sum = 0
        accuracy = 0
        w_gradient = [np.zeros_like(layer) for layer in w]
        b_gradient = [np.zeros_like(layer) for layer in b]

        for image_idx in range(first_sample, first_sample + batch_size):
            image_idx %= len(global_values.x_train)
            y = global_values.y_train_one_hot[image_idx]
            # image = np.array(global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)).reshape(28, 28)
            # a = [image_processor.extract_feature(image)] + [None] * len(w)

            a = [global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)] + [None] * len(w)
            a, z = forward_pass(len(layers), a, w, b)

            costs_sum += np.sum((a[-1] - y) ** 2)
            if np.argmax(a[-1]) == global_values.y_train[image_idx]:
                accuracy += 1

            cost_z = [None] * len(w)

            cost_z[-1] = 2 * (a[-1] - y) * global_values.df(z[-1]) * 2
            b_gradient[-1] += cost_z[-1]
            w_gradient[-1] += np.outer(cost_z[-1], a[-2])

            for l in range(len(w) - 2, -1, -1):
                cost_z[l] = np.dot(w[l + 1].T, cost_z[l + 1]) * global_values.df(z[l]) * 2
                b_gradient[l] += cost_z[l]
                w_gradient[l] += np.outer(cost_z[l], a[l])

        for l in range(len(w)):
            w[l] -= learning_rate * w_gradient[l] / batch_size
            b[l] -= learning_rate * b_gradient[l] / batch_size

        cost_series[x] = costs_sum / layers[-1] / batch_size
        accuracy_series[x] = accuracy / batch_size

        # print(f'{x}# cost : {cost_series[x]}, accuracy : {accuracy_series[x]}')

    print(time.time() - start)

    global_values.NN_list.append({
        'layers': layers,
        'iterations': batches,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'noise': noise,
        'cost': cost_series,
        'accuracy': accuracy_series,
        'b': b,
        'w': w
    })