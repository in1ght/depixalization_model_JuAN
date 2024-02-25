import numpy as np
import torch
import math
from torchvision import transforms
from matplotlib import pyplot as plt

def stack_with_padding(target,pixelated_image,known_arr, amount, mean_std):
    # Expected list elements are 3-tuples:
    # (pixelated_image, known_array, target_array)
    n = amount
    pixelated_images_dtype = "float32"  # Same for every sample
    known_arrays_dtype = known_arr[0].dtype
    shapes = []
    pixelated_images = []
    known_arrays = []
    target_arrays = []
    
    for target_array,pixelated_image, known_array, mean_std in zip(target,pixelated_image,known_arr,mean_std):
        shapes.append(pixelated_image.shape)  # Equal to known_array.shape

        target_arrays.append(target_array)
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
        

    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.zeros(shape=(n, *max_shape), dtype=known_arrays_dtype)

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]

    stacked_known_arrays = torch.from_numpy(stacked_known_arrays)
    stacked_pixelated_images = torch.from_numpy(stacked_pixelated_images)



    combined_arr = np.zeros(shape=(n, 2,64,64), dtype=np.float32)

    for i in range(n):
        combined_arr[i] = torch.cat([stacked_pixelated_images[i], ~(stacked_known_arrays[i])])

    return torch.from_numpy(combined_arr), stacked_known_arrays, torch.from_numpy(np.array(target_arrays, dtype = np.float32))





def stack_with_padding_just_for_test(pixelated_image,known_arr, amount):
    n = amount
    pixelated_images_dtype = np.float32
    shapes = []
    pixelated_images = []
    known_arrays = []
    
    for pixelated_image, known_array in zip(pixelated_image,known_arr):
        shapes.append(pixelated_image.shape)
        pixelated_images.append(pixelated_image)
        known_arrays.append(known_array)
    
    max_shape = np.max(np.stack(shapes, axis=0), axis=0)
    stacked_pixelated_images = np.zeros(shape=(n, *max_shape), dtype=pixelated_images_dtype)
    stacked_known_arrays = np.ones(shape=(n, *max_shape), dtype=bool)

    for i in range(n):
        channels, height, width = pixelated_images[i].shape
        stacked_pixelated_images[i, :channels, :height, :width] = pixelated_images[i]
        stacked_known_arrays[i, :channels, :height, :width] = known_arrays[i]

    stacked_known_arrays = torch.from_numpy(stacked_known_arrays)
    stacked_pixelated_images = torch.from_numpy(stacked_pixelated_images)

    combined_arr = np.zeros(shape=(n, 2,64,64), dtype="float32")

    for i in range(n):
        combined_arr[i] = torch.cat([stacked_pixelated_images[i], (stacked_known_arrays[i])])

    return torch.from_numpy(combined_arr), ~stacked_known_arrays