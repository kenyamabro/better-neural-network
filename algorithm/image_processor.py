import numpy as np
import time
import global_values

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
    wider = w_over_h > global_values.avg_w_over_h

    # Add black strips to larger side
    side = width if wider else height
    strip1 = int(global_values.strip_ratios[2 * int(not wider)] * side)
    strip2 = int(global_values.strip_ratios[2 * int(not wider) + 1] * side)
    pixels = add_black_strips(pixels, strip1, wider)
    pixels = add_black_strips(pixels, strip2, wider)

    # Add strips to form a square
    strip = (pixels.shape[int(wider)] - pixels.shape[int(not wider)]) // 2
    pixels = add_black_strips(pixels, strip, not wider)

    return pixels

def pixelize_image(pixels):
    pixelized_image = []
    grid_size = pixels.shape[0] / 28

    for row in range(28):
        for col in range(28):
            x_start = int(col * grid_size)
            x_end = int((col + 1) * grid_size)
            y_start = int(row * grid_size)
            y_end = int((row + 1) * grid_size)

            cell_pixels = pixels[y_start:y_end, x_start:x_end]

            avg_color = np.mean(cell_pixels, axis=(0, 1))

            pixelized_image.append(avg_color[0])

    return pixelized_image

def extract_feature(initial_image):
    images = np.array([initial_image])
    for kernels, biases, AF in zip(global_values.kernels_w, global_values.kernels_b, global_values.AFs_list):
        images = convolve(images, kernels, biases, 1, AF)
        images = np.array([max_pool(image, 2, 2) for image in images])
    return images.flatten()

def convolve(images, kernels, biases, stride, AF):
    kernels = np.array(kernels)
    convolved_images = []
    for kernel, bias in zip(kernels, biases):
        kernel_height, kernel_width = kernel.shape[1], kernel.shape[2]
        output_height = (images.shape[1] - kernel_height) // stride + 1
        output_width = (images.shape[2] - kernel_width) // stride + 1
        convolved_image = np.zeros((output_height, output_width))

        for y in range(0, images.shape[1] - kernel_height + 1, stride):
            for x in range(0, images.shape[2] - kernel_width + 1, stride):
                region = images[:, y:y+kernel_height, x:x+kernel_width]
                convolved_image[y // stride, x // stride] = AF(np.sum(region * kernel) + bias)

        convolved_images.append(convolved_image)

    return np.array(convolved_images)

def max_pool(image, pool_size, stride):
    out_height = (image.shape[0] - pool_size) // stride + 1
    out_width = (image.shape[1] - pool_size) // stride + 1

    windows = np.lib.stride_tricks.sliding_window_view(image, (pool_size, pool_size))
    windows = windows[::stride, ::stride]

    return windows.reshape(out_height, out_width, -1).max(axis=2)

# test_image = np.random.uniform(0, 1, (28, 28))
# # start_time = time.time()
# reduced_image = extract_feature(test_image)
# # end_time = time.time()
# # print(end_time - start_time)