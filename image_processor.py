import numpy as np
import network

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