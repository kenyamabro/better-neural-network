import tkinter as tk
from tkinter import ttk
import numpy as np
from PIL import ImageGrab
from screeninfo import get_monitors
import inflect
import network

monitors = get_monitors()
ie = inflect.engine()

def add_black_strips(image_array, strip_thickness, wider):
    if wider: # Vertical strips
        strip = np.zeros((image_array.shape[0], strip_thickness, image_array.shape[2]), dtype=image_array.dtype)
        return np.hstack((strip, image_array, strip))
    else: # Horizontal strips
        strip = np.zeros((strip_thickness, image_array.shape[1], image_array.shape[2]), dtype=image_array.dtype)
        return np.vstack((strip, image_array, strip))
    
def center_image(pixels):
    colored_area = np.argwhere(pixels[:, :, :3].any(axis=-1))
    y_start, x_start = colored_area.min(axis=0)
    y_end, x_end = colored_area.max(axis=0)
    pixels = pixels[y_start:y_end + 1, x_start:x_end + 1]

    width = x_end - x_start
    height = y_end - y_start
    w_over_h = width / height
    wider = w_over_h > network.avg_w_over_h

    # Add black strips to larger side
    side = width if wider else height
    strip1 = int(network.strip_ratios[2 * int(not wider)] * side)
    strip2 = int(network.strip_ratios[2 * int(not wider) + 1] * side)
    pixels = add_black_strips(pixels, strip1, wider)
    pixels = add_black_strips(pixels, strip2, wider)

    # Add strips to form a square
    strip = (pixels.shape[int(wider)] - pixels.shape[int(not wider)]) // 2
    pixels = add_black_strips(pixels, strip, not wider)

    return pixels

def create_drawing_interface(NNid, NN, parameters_info):
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

        canvas_bbox = (canvas_x, canvas_y, canvas_x + canvas_width, canvas_y + canvas_height)

        image = ImageGrab.grab(bbox=canvas_bbox)
        pixels = np.array(image)
        pixels = pixels[3:-2, 3:-2] # Cut white borders

        pixels = center_image(pixels)
        grid_size = pixels.shape[0] / 28
        r = grid_size * 28 / grid_canvas.winfo_width()

        grid_canvas.delete("all")
        pixelized_image = []

        for row in range(28):
            for col in range(28):
                x_start = int(col * grid_size)
                x_end = int((col + 1) * grid_size)
                y_start = int(row * grid_size)
                y_end = int((row + 1) * grid_size)

                cell_pixels = pixels[y_start:y_end, x_start:x_end]

                avg_color = np.mean(cell_pixels, axis=(0, 1))

                avg_color = tuple(int(c) for c in avg_color)
                color_hex = f'#{avg_color[0]:02x}{avg_color[1]:02x}{avg_color[2]:02x}'
                grid_canvas.create_rectangle(
                    x_start // r, y_start // r, x_end // r, y_end // r,
                    fill=color_hex, outline=color_hex
                )

                pixelized_image.append(avg_color[0])

        pixelized_image = np.array(pixelized_image).reshape(28, 28)
        colored_area = np.argwhere(pixelized_image != 0)
        x_start, y_start = colored_area.min(axis=0)
        x_end, y_end = colored_area.max(axis=0)

        pixelized_image = pixelized_image.flatten()
        a = [np.array(pixelized_image) / 255.0]
        a, z = network.forward_pass(a, len(NN['hidden_layers']) + 2, NN['w'], NN['b'])
        sorted_costs_indices = np.argsort(a[-1])[::-1]

        guesses_listbox.delete(0, tk.END)
        for rank, i in enumerate(sorted_costs_indices):
            guesses_listbox.insert(rank, f'#{rank+1} : {ie.number_to_words(i)} (activation : {a[-1][i]})')

    drawing_window = tk.Toplevel()
    drawing_window.title(f'[{NNid}] Drawing test')

    canvas_layout = tk.Frame(drawing_window, width=400)
    canvas_layout.grid(row=0, column=0, sticky='n')

    tk.Label(canvas_layout, text=f"{parameters_info}\nTest the neural network by drawing single digits:", anchor='w').grid(row=0, column=0, sticky='ew')
    canvas = tk.Canvas(canvas_layout, width=400, height=400, bg='black')
    canvas.grid(row=1, column=0)
    canvas.bind("<B1-Motion>", draw)

    separator = ttk.Separator(drawing_window, orient='vertical')
    separator.grid(row=0, column=1, sticky='ns')

    control_layout = tk.Frame(drawing_window, width=200)
    control_layout.grid(row=0, column=2, sticky='n')

    tk.Label(control_layout, text='font size:', anchor='w').grid(row=0, column=0, sticky='w')
    font_size_entry = tk.Entry(control_layout)
    font_size_entry.grid(row=0, column=1, sticky='ew')
    font_size_entry.insert(0, '15')

    tk.Label(control_layout, text='color:', anchor='w').grid(row=1, column=0, sticky='w')
    color_var = tk.StringVar(value='white')
    color_menu = tk.OptionMenu(control_layout, color_var, 'black', 'white')
    color_menu.grid(row=1, column=1, sticky='ew')

    clear_button = tk.Button(control_layout, text='Clear', command=clear_canvas)
    clear_button.grid(row=2, column=0, sticky='ew')

    read_button = tk.Button(control_layout, text='Read', command=read_canvas)
    read_button.grid(row=2, column=1, sticky='ew')

    tk.Label(control_layout, text='read image:\n(resized and\ncentered for\nmore accuracy,\nbut not\nreshaped!)', anchor='w').grid(row=3, column=0, sticky='nw')
    grid_canvas = tk.Canvas(control_layout, width=200, height=200, bg='black')
    grid_canvas.grid(row=3, column=1)

    tk.Label(control_layout, text='guesses:', anchor='w').grid(row=4, column=0, sticky='nw')
    guesses_listbox = tk.Listbox(control_layout, height=4, width=15)
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