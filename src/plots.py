import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

def create_weights_heatmap(NNid, NN: NeuralNetwork, parameters_info):
    map_num = NN.layers[1]
    rows = np.floor(np.sqrt(map_num)-0.0001).astype('int')
    columns = rows + 1
    if rows * (columns) < map_num:
        rows += 1

    plt.figure(figsize=((5 / rows) * columns, 5))
    plt_title = 'Heatmaps of Weights between Each Second-Layer and All Input Neurons'
    plt.get_current_fig_manager().set_window_title(f'[{NNid}] {plt_title}')
    plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

    max_weight = np.max(np.abs(NN.w[0]))
    side = int(np.sqrt(NN.w[0].shape[1]))

    for i in range(map_num):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(NN.w[0][i].reshape(side, side), cmap='bwr', aspect='auto', vmin=-max_weight, vmax=max_weight)
        if i + 1 == map_num: plt.colorbar()
        plt.axis('off')

    plt.tight_layout()
    plt.show(block=False)

def create_cost_acuracy_plot(NNid, NN: NeuralNetwork, parameters_info):
    plt.figure(figsize=(5, 5))
    plt_title = 'Cost and Accuracy vs. Batch Number'
    plt.get_current_fig_manager().set_window_title(f'[{NNid}] {plt_title}')
    plt.suptitle(f'{plt_title}\n{parameters_info}', fontsize=10)

    cost_line, = plt.plot(NN.cost_series, label='Cost per output neuron', color='orange', marker='o', markersize=2, linewidth=1)
    accuracy_line, = plt.plot(NN.accuracy_series, label='Accuracy', color='green', marker='o', markersize=2, linewidth=1)

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