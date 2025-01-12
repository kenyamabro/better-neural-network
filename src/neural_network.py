import numpy as np
import image_processor
import time
import global_values

class NeuralNetwork:
    NN_list = []

    def __init__(self, hidden_layers, batches, batch_size, learning_rate, noise):
        self.layers = [784] + hidden_layers + [10]
        self.batches = batches
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.noise = noise

        self.w = [np.random.uniform(-1, 1, (self.layers[i + 1], self.layers[i]))
                  for i in range(len(self.layers) - 1)]
        self.b = [np.random.uniform(-0.5, 0.5, self.layers[i + 1])
                  for i in range(len(self.layers) - 1)]

        start = time.time()

        self.cost_series = np.zeros(batches)
        self.accuracy_series = np.zeros(batches)

        f = global_values.f
        df = global_values.df

        for x in range(batches):
            first_sample = batch_size * x
            costs_sum = 0
            accuracy = 0
            w_gradient = [np.zeros_like(layer) for layer in self.w]
            b_gradient = [np.zeros_like(layer) for layer in self.b]

            for image_idx in range(first_sample, first_sample + batch_size):
                image_idx %= len(global_values.x_train)
                y = global_values.y_train_one_hot[image_idx]
                # image = np.array(global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)).reshape(28, 28)
                # a = [image_processor.extract_feature(image)] + [None] * len(w)

                a = [global_values.x_train[image_idx] + np.random.uniform(-noise, noise, 784)] + [None] * len(self.w)
                a, z = self.forward_pass(a, f)

                costs_sum += np.sum((a[-1] - y) ** 2)
                if np.argmax(a[-1]) == global_values.y_train[image_idx]:
                    accuracy += 1

                cost_z = [None] * len(self.w)

                cost_z[-1] = 2 * (a[-1] - y) * df(z[-1]) * 2
                b_gradient[-1] += cost_z[-1]
                w_gradient[-1] += np.outer(cost_z[-1], a[-2])

                for l in range(len(self.w) - 2, -1, -1):
                    cost_z[l] = np.dot(self.w[l + 1].T, cost_z[l + 1]) * df(z[l]) * 2
                    b_gradient[l] += cost_z[l]
                    w_gradient[l] += np.outer(cost_z[l], a[l])

            for l in range(len(self.w)):
                self.w[l] -= learning_rate * w_gradient[l] / batch_size
                self.b[l] -= learning_rate * b_gradient[l] / batch_size

            self.cost_series[x] = costs_sum / self.layers[-1] / batch_size
            self.accuracy_series[x] = accuracy / batch_size

        print(time.time() - start)
        NeuralNetwork.NN_list.append(self)

    def forward_pass(self, a, f):
        z = [None] * (len(self.layers) - 1)
        for l in range(len(self.layers) - 1):
            z[l] = np.dot(self.w[l], a[l]) + self.b[l]
            a[l + 1] = f(z[l])
        return a, z