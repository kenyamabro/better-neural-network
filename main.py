import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageGrab
from screeninfo import get_monitors
import inflect

monitors = get_monitors()
ie = inflect.engine()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 784).astype('float32') / 255.0
y_train_one_hot = np.eye(10)[y_train]

NN_list = []

def forward_pass(a, layers_num, w, b):
    z = []
    for l in range(1, layers_num):
        z.append(np.dot(w[l - 1], a[l - 1]) + b[l - 1])
        a.append((np.tanh(z[-1]) + 1) / 2)
    return a, z

def create_network(hidden_layers, batch_size, learning_rate, noise):
    layers = [784] + hidden_layers + [10]

    global w, b, NN_list
    w = [np.random.uniform(-1, 1, (layers[i + 1], layers[i]))
         for i in range(len(layers) - 1)]
    b = [np.random.uniform(-0.5, 0.5, layers[i + 1])
         for i in range(len(layers) - 1)]

    def sech(x):
        return 2 / (np.exp(x) + np.exp(-x))

    def minimize_cost_function():
        cost_series = []
        accuracy_series = []
        for x in range(len(x_train) // batch_size):
            first_sample = batch_size * x
            costs_sum = 0
            accuracy = 0
            w_gradient = [np.zeros_like(layer) for layer in w]
            b_gradient = [np.zeros_like(layer) for layer in b]

            for image_idx in range(first_sample, first_sample + batch_size):
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
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'noise': noise,
        'cost': cost_series,
        'accuracy': accuracy_series,
        'b': b,
        'w': w
    })

    NNid = len(NN_list)
    listbox.insert(NNid, f'#{NNid} : {hidden_layers}, {batch_size}, {learning_rate}, {noise}')

def run_training():
    try:
        hidden_layers = list(map(int, entries[0].get().split(',')))
        batch_size = int(entries[1].get())
        learning_rate = float(entries[2].get())
        noise = float(entries[3].get())

        train_text_var.set('Training in Progress...')
        root.after(100, start_training, hidden_layers, batch_size, learning_rate, noise)
    except ValueError as e:
        print(f'ValueError: {e}')
        messagebox.showerror('Input Error', f'Please enter valid numbers. Error: {e}')
    except Exception as e:
        print(f'Exception: {e}')
        messagebox.showerror('Unexpected Error', f'An unexpected error occurred. Error: {e}')

def start_training(hidden_layers, batch_size, learning_rate, noise):
    create_network(hidden_layers, batch_size, learning_rate, noise)
    train_text_var.set('')
    placeholder_label.lower()

def create_entry(text, row, default_entry):
    tk.Label(root, text=text, anchor='w').grid(row=row, column=0, sticky='w')
    entry = tk.Entry(root)
    entry.grid(row=row, column=1)
    entry.insert(0, default_entry)
    entries.append(entry)

def on_item_select(event):
    widget = event.widget
    NNid = widget.nearest(event.y)

    if NNid >= 0:
        NN = NN_list[NNid]
        NNid += 1
        hidden_layers = NN['hidden_layers']
        parameters_info = f'Hidden layers: {hidden_layers}, Batch size: {NN['batch_size']}, Learning rate: {NN['learning_rate']}, Noise: {NN['noise']}'

        map_num = hidden_layers[0]
        rows = np.floor(np.sqrt(map_num)-0.0001).astype('int')
        columns = rows + 1
        if rows * (columns) < map_num:
            rows += 1

        plt.figure(figsize=((5 / rows) * columns, 5))
        plt_title = 'Heatmaps of Weights between Each Second-Layer and All Input Neurons'
        plt.get_current_fig_manager().set_window_title(f'[{NNid}] {plt_title}')
        plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

        max_weight = np.max(np.abs(w[0]))

        for i in range(map_num):
            plt.subplot(rows, columns, i + 1)
            plt.imshow(NN['w'][0][i].reshape(28, 28), cmap='bwr', aspect='auto', vmin=-max_weight, vmax=max_weight)
            if i + 1 == map_num: plt.colorbar()
            plt.axis('off')

        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(5, 5))
        plt_title = 'Cost and Accuracy vs. Batch Number'
        plt.get_current_fig_manager().set_window_title(f'[{NNid}] {plt_title}')
        plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

        cost_line, = plt.plot(NN['cost'], label='Cost per output neuron', color='orange', marker='o', markersize=2, linewidth=1)
        accuracy_line, = plt.plot(NN['accuracy'], label='Accuracy', color='green', marker='o', markersize=2, linewidth=1)

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

        def draw(event):
            x, y = event.x, event.y
            try: fs = int(font_size_entry.get())
            except: return
            color = color_var.get()
            canvas.create_oval(x-fs, y-fs, x+fs, y+fs, fill=color, outline=color)

        def clear_canvas():
            canvas.delete('all')

        def read_canvas():
            guesses_placeholder_label.lower()

            canvas_x = canvas.winfo_rootx()
            canvas_y = canvas.winfo_rooty()
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()

            r = canvas_width / grid_canvas.winfo_width()

            canvas_bbox = (canvas_x, canvas_y, canvas_x + canvas_width, canvas_y + canvas_height)

            image = ImageGrab.grab(bbox=canvas_bbox)
            pixels = np.array(image)

            grid_canvas.delete("all")

            grid_size_x = canvas_width // 28
            grid_size_y = canvas_height // 28

            pixelized_image = []

            for row in range(28):
                for col in range(28):
                    x_start = col * grid_size_x
                    x_end = (col + 1) * grid_size_x
                    y_start = row * grid_size_y
                    y_end = (row + 1) * grid_size_y

                    cell_pixels = pixels[y_start:y_end, x_start:x_end]

                    avg_color = np.mean(cell_pixels, axis=(0, 1))

                    avg_color = tuple(int(c) for c in avg_color)
                    color_hex = f'#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}'
                    pixelized_image.append(avg_color[0])

                    grid_canvas.create_rectangle(
                        x_start // r, y_start // r, x_end // r, y_end // r,
                        fill=color_hex, outline=color_hex
                    )

            a = [np.array(pixelized_image) / -255.0 + 1]
            a, z = forward_pass(a, len(hidden_layers) + 2, NN['w'], NN['b'])
            sorted_costs_indices = np.argsort(a[-1])[::-1]

            guesses_listbox.delete(0, tk.END)
            for rank, i in enumerate(sorted_costs_indices):
                guesses_listbox.insert(rank, f'#{rank+1} : {ie.number_to_words(i)} (cost : {a[-1][i]})')

        drawing_window = tk.Toplevel()
        drawing_window.title(f'[{NNid}] Drawing test')

        canvas_layout = tk.Frame(drawing_window, width=400)
        canvas_layout.grid(row=0, column=0, sticky='n')

        tk.Label(canvas_layout, text=f'{parameters_info}\nTest the neural network by drawing single digits:', anchor='w').grid(row=0, column=0, sticky='ew')
        canvas = tk.Canvas(canvas_layout, width=400, height=400, bg='white')
        canvas.grid(row=1, column=0)
        canvas.bind("<B1-Motion>", draw)

        control_layout = tk.Frame(drawing_window, width=200)
        control_layout.grid(row=0, column=1, sticky='n')

        tk.Label(control_layout, text='font size:', anchor='w').grid(row=0, column=0, sticky='w')
        font_size_entry = tk.Entry(control_layout)
        font_size_entry.grid(row=0, column=1, sticky='ew')
        font_size_entry.insert(0, '10')

        tk.Label(control_layout, text='color:', anchor='w').grid(row=1, column=0, sticky='w')
        color_var = tk.StringVar(value='black')
        color_menu = tk.OptionMenu(control_layout, color_var, 'black', 'white')
        color_menu.grid(row=1, column=1, sticky='ew')

        clear_button = tk.Button(control_layout, text='Clear', command=clear_canvas)
        clear_button.grid(row=2, column=0, sticky='ew')

        read_button = tk.Button(control_layout, text='Read', command=read_canvas)
        read_button.grid(row=2, column=1, sticky='ew')

        tk.Label(control_layout, text='read image:', anchor='w').grid(row=3, column=0, sticky='nw')
        grid_canvas = tk.Canvas(control_layout, width=200, height=200, bg='white')
        grid_canvas.grid(row=3, column=1)

        tk.Label(control_layout, text='guesses:', anchor='w').grid(row=4, column=0, sticky='nw')
        guesses_listbox = tk.Listbox(control_layout, height=4)
        guesses_listbox.grid(row=4, column=1, sticky='ew')
        guesses_placeholder_label = tk.Label(control_layout, text="Read your drawing first", fg="gray", bg="white")
        control_layout.after(100,
            lambda: guesses_placeholder_label.place(
                x=guesses_listbox.winfo_x(),
                y=guesses_listbox.winfo_y(),
                width=guesses_listbox.winfo_width(),
                height=guesses_listbox.winfo_height()
            )
        )

root = tk.Tk()
root.title('Neural Network Trainer')
entries = []

tk.Label(root, text='Dataset used:', anchor='w').grid(row=0, column=0, sticky='nw')
tk.Label(root, text='MNIST dataset\n (28x28 pixels images of\n handwritten single digits)', anchor='w').grid(row=0, column=1, sticky='w')

tk.Label(root, text='Datapoints:', anchor='w').grid(row=1, column=0, sticky='w')
tk.Label(root, text=len(x_train), anchor='w').grid(row=1, column=1, sticky='w')

create_entry('Hidden Layers (comma-separated):', 2, '20,20')
create_entry('Batch Size:', 3, '50')
create_entry('Learning Rate:', 4, '0.4')
create_entry('Noise:', 5, '0')

train_button = tk.Button(root, text='Train', command=run_training)
train_button.grid(row=6, column=0, sticky='ew')
train_text_var = tk.StringVar(value='')
tk.Label(root, textvariable=train_text_var, anchor='w').grid(row=6, column=1, sticky='w')

listbox = tk.Listbox(root)
listbox.grid(row=7, column=0, columnspan=2, sticky='ew')
listbox.bind("<Double-Button-1>", on_item_select)

placeholder_label = tk.Label(root, text="The trained neural networks will appear here.\nDouble click to open them.", fg="gray", bg="white")
root.after(100,
    lambda: placeholder_label.place(
        x=listbox.winfo_x(),
        y=listbox.winfo_y(),
        width=listbox.winfo_width(),
        height=listbox.winfo_height()
    )
)

root.mainloop()