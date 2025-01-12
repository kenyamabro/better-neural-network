import tkinter as tk
from tkinter import messagebox
from neural_network import NeuralNetwork
import plots
import drawing_interface
import global_values
import time

def run_training():
    try:
        hidden_layers = list(map(int, entries[0].get().split(',')))
        batches = int(entries[1].get())
        batch_size = int(entries[2].get())
        learning_rate = float(entries[3].get())
        noise = float(entries[4].get())

        train_text_var.set('Training in Progress...')
        root.after(100, start_training, hidden_layers, batches, batch_size, learning_rate, noise)
    except ValueError as e:
        print(f'ValueError: {e}')
        messagebox.showerror('Input Error', f'Please enter valid numbers. Error: {e}')
    except Exception as e:
        print(f'Exception: {e}')
        messagebox.showerror('Unexpected Error', f'An unexpected error occurred. Error: {e}')

def start_training(hidden_layers, batches, batch_size, learning_rate, noise):
    start = time.time()
    nn = NeuralNetwork(hidden_layers, batches, batch_size, learning_rate, noise)
    print(time.time() - start)

    train_text_var.set('')
    placeholder_label.lower()
    listbox.insert(len(NeuralNetwork.NN_list), f'#{len(NeuralNetwork.NN_list)} : {hidden_layers}, {batches}, {batch_size}, {learning_rate}, {noise}')

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
        NN: NeuralNetwork = NeuralNetwork.NN_list[NNid]
        NNid += 1
        parameters_info = f'Hidden layers: {NN.layers[1:-1]}, Batches: {NN.batches} Batch size: {NN.batch_size},\nLearning rate: {NN.learning_rate}, Noise: {NN.noise}'

        plots.create_weights_heatmap(NNid, NN, parameters_info)
        plots.create_cost_acuracy_plot(NNid, NN, parameters_info)
        drawing_interface.create_drawing_interface(NNid, NN, parameters_info)

root = tk.Tk()
root.title('Neural Network Trainer')
entries = []

tk.Label(root, text='Dataset used:', anchor='w').grid(row=0, column=0, sticky='nw')
tk.Label(root, text='MNIST dataset\n (28x28 pixels images of\n handwritten single digits)', anchor='w').grid(row=0, column=1, sticky='w')

tk.Label(root, text='Datapoints:', anchor='w').grid(row=1, column=0, sticky='w')
tk.Label(root, text=len(global_values.x_train), anchor='w').grid(row=1, column=1, sticky='w')

create_entry('Hidden Layers (Comma-Separated):', 2, '10,20')
create_entry('Batches:', 3, '1200')
create_entry('Batch Size:', 4, '50')
create_entry('Learning Rate:', 5, '0.4')
create_entry('Noise:', 6, '0')

train_button = tk.Button(root, text='Train', command=run_training)
train_button.grid(row=7, column=0, sticky='ew')
train_text_var = tk.StringVar(value='')
tk.Label(root, textvariable=train_text_var, anchor='w').grid(row=7, column=1, sticky='w')

tk.Label(root, text='Double click to open the neural networks:', anchor='w').grid(row=8, column=0, columnspan=2, sticky='w')

listbox = tk.Listbox(root)
listbox.grid(row=9, column=0, columnspan=2, sticky='ew')
listbox.bind("<Double-Button-1>", on_item_select)

placeholder_label = tk.Label(root, text="The trained neural networks will appear here.", fg="gray", bg="white")
root.after(100,
    lambda: placeholder_label.place(
        x=listbox.winfo_x(),
        y=listbox.winfo_y(),
        width=listbox.winfo_width(),
        height=listbox.winfo_height()
    )
)

root.mainloop()