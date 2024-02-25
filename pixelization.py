import numpy as np

def pixelization(image_place: np.ndarray,rseed: int) -> tuple:
    image = np.copy(image_place)
    rng = np.random.default_rng(rseed) #seed

    p_width = rng.integers(4, 33)
    p_height = rng.integers(4, 33)

    x = rng.integers(0, 64 - p_width)
    y = rng.integers(0, 64 - p_height)
    size = rng.integers(4, 17)

    area = (..., slice(y, y + p_height), slice(x, x + p_width))
    
    known_array = np.zeros_like(image, dtype=bool)
    known_array[area] = True

    current_x = x

    while current_x < x + p_width:
        current_y = y
        while current_y < y + p_height:
            block = (..., slice(current_y, min(current_y + size, y + p_height)), slice(current_x, min(current_x + size, x + p_width)))
            image[block] = image[block].mean()
            current_y += size
        current_x += size
    
    return image, known_array

