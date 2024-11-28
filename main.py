import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
y_train_one_hot = np.eye(10)[y_train]

NNid = 0

def create_network(hidden_layers, batch_size, learning_rate, noise):
    layers = [784] + hidden_layers + [10]

    global w, b, NNid
    w = [np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
         for i in range(len(layers) - 1)]
    b = [np.random.uniform(-0.5, 0.5, layers[i + 1])
         for i in range(len(layers) - 1)]

    def sech(x):
        return 2 / (np.exp(x) + np.exp(-x))

    def minimize_cost_function(first_sample):
        costs_sum = 0
        accuracy = 0
        w_gradient = [np.zeros_like(layer) for layer in w]
        b_gradient = [np.zeros_like(layer) for layer in b]

        for image_idx in range(first_sample, first_sample + batch_size):
            y = y_train_one_hot[image_idx]
            a = [x_train[image_idx] + np.random.uniform(-noise, noise, 784)]

            z = []
            for l in range(1, len(layers)):
                z.append(np.dot(w[l - 1], a[l - 1]) + b[l - 1])
                a.append((np.tanh(z[-1]) + 1) / 2)

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
        return cost, accuracy

    cost_series = []
    accuracy_series = []
    for x in range(len(x_train) // batch_size):
        cost, accuracy = minimize_cost_function(batch_size * x)
        cost_series.append(cost)
        accuracy_series.append(accuracy)

    NNid += 1
    parameters_info = f'Hidden layers: {hidden_layers}, Batch size: {batch_size}, Learning rate: {learning_rate}, Noise: {noise}'

    plt.figure(figsize=(5, 5))
    plt_title = 'Cost and Accuracy vs. Batch Number'
    plt.get_current_fig_manager().set_window_title(f"[{NNid}] {plt_title}")
    plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

    cost_line, = plt.plot(cost_series, label='Cost per output neuron', color='blue', marker='o', markersize=2, linewidth=1)
    accuracy_line, = plt.plot(accuracy_series, label='Accuracy', color='green', marker='o', markersize=2, linewidth=1)

    plt.xlabel('Batch Number')
    plt.ylabel('Cost and Accuracy')
    plt.grid(True)

    legend = plt.legend()
    for legend_line, original_line in zip(legend.get_lines(), [cost_line, accuracy_line]):
        legend_line.original_line = original_line
        legend_line.set_picker(True)

    def on_legend_click(event):
        legend_line = event.artist
        original_line = legend_line.original_line
        
        visible = not original_line.get_visible()
        original_line.set_visible(visible)
        
        legend_line.set_alpha(1.0 if visible else 0.2)
        plt.gcf().canvas.draw()

    plt.gcf().canvas.mpl_connect('pick_event', on_legend_click)

    plt.tight_layout()
    plt.show(block=False)

    map_num = layers[1]
    rows = np.floor(np.sqrt(map_num)-0.0001).astype("int")
    columns = rows + 1
    if rows * (columns) < map_num:
        rows += 1

    plt.figure(figsize=((5 / rows) * columns, 5))
    plt_title = 'Heatmaps of Weights between Each Second-Layer and All Input Neurons'
    plt.get_current_fig_manager().set_window_title(f"[{NNid}] {plt_title}")
    plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

    max_weight = np.max(np.abs(w[0]))

    for i in range(map_num):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(w[0][i].reshape(28, 28), cmap='bwr', aspect='auto', vmin=-max_weight, vmax=max_weight)
        if i + 1 == map_num: plt.colorbar()
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def run_training():
    try:
        hidden_layers = list(map(int, entries[0].get().split(',')))
        batch_size = int(entries[1].get())
        learning_rate = float(entries[2].get())
        noise = float(entries[3].get())

        create_network(hidden_layers, batch_size, learning_rate, noise)
    except ValueError as e:
        print(f"ValueError: {e}")
        messagebox.showerror("Input Error", f"Please enter valid numbers. Error: {e}")
    except Exception as e:
        print(f"Exception: {e}")
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred. Error: {e}")

root = tk.Tk()
root.title("Parameters Entry")
entries = []

def create_entry(text, row, default_entry):
    tk.Label(root, text=text, anchor='w').grid(row=row, column=0, sticky='w')
    entry = tk.Entry(root)
    entry.grid(row=row, column=1)
    entry.insert(0, default_entry)
    entries.append(entry)

create_entry("Hidden Layers (comma-separated):", 0, "20,20")
create_entry("Batch Size:", 1, "50")
create_entry("Learning Rate:", 2, "0.4")
create_entry("Noise:", 3, "0")

train_button = tk.Button(root, text="Train", command=run_training)
train_button.grid(row=4, column=0, columnspan=2)

root.mainloop()