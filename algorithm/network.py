import numpy as np
import algorithm.image_processor as image_processor
import time
import global_values

k = 2
AF = global_values.AFs('logistic', k = k)

def forward_pass(a, layers_num, w, b, AF):
    z = []
    for l in range(1, layers_num):
        z.append(np.dot(w[l - 1], a[l - 1]) + b[l - 1])
        a.append(AF(z[-1]))
    return a, z

def create_network(hidden_layers, batches, batch_size, learning_rate, noise):
    layers = [784] + hidden_layers + [10]
    # layers = [global_values.inputs_num] + hidden_layers + [10]

    global w, b
    w = [np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
         for i in range(len(layers) - 1)]
    b = [np.random.uniform(-0.5, 0.5, layers[i + 1])
         for i in range(len(layers) - 1)]

    def minimize_cost_function():
        cost_series = []
        accuracy_series = []
        for x in range(batches):
            first_sample = batch_size * x
            costs_sum = 0
            accuracy = 0
            w_gradient = [np.zeros_like(layer) for layer in w]
            b_gradient = [np.zeros_like(layer) for layer in b]
            # runtime = 0

            for image_idx in range(first_sample, first_sample + batch_size):
                image_idx %= len(global_values.x_train)
                y = global_values.y_train_one_hot[image_idx]
                # image = np.array(global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)).reshape(28, 28)
                # # start_time = time.time()
                # a = [image_processor.extract_feature(image)]
                # # end_time = time.time()
                # # runtime += end_time - start_time

                a = [global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)]
                a, z = forward_pass(a, len(layers), w, b, AF)

                costs_sum += np.sum((a[-1] - y) ** 2)
                if np.argmax(a[-1]) == global_values.y_train[image_idx]:
                    accuracy += 1

                cost_z = [None] * len(w)

                cost_z[-1] = 2 * (a[-1] - y) * AF(z[-1], True) * k
                b_gradient[-1] += cost_z[-1]
                w_gradient[-1] += np.outer(cost_z[-1], a[-2])

                for l in range(len(w) - 2, -1, -1):
                    cost_z[l] = np.dot(w[l + 1].T, cost_z[l + 1]) * AF(z[l], True) * k
                    b_gradient[l] += cost_z[l]
                    w_gradient[l] += np.outer(cost_z[l], a[l])

            for l in range(len(w)):
                w[l] -= learning_rate * w_gradient[l] / batch_size
                b[l] -= learning_rate * b_gradient[l] / batch_size

            cost = costs_sum / layers[-1] / batch_size
            accuracy /= batch_size

            cost_series.append(cost)
            accuracy_series.append(accuracy)

            # print(f'{x}# time : {runtime}, cost : {cost}, accuracy : {accuracy}')
            print(f'{x}# cost : {cost}, accuracy : {accuracy}')

        return cost_series, accuracy_series

    cost_series, accuracy_series = minimize_cost_function()

    global_values.NN_list.append({
        'hidden_layers': hidden_layers,
        'iterations': batches,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'noise': noise,
        'cost': cost_series,
        'accuracy': accuracy_series,
        'b': b,
        'w': w
    })