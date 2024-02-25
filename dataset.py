import torch
from torch.utils.data import random_split, TensorDataset

from padding import stack_with_padding

def get_dataset(target,pixelated,known_arr,amount,mean_std):
    
    combined,known,target = stack_with_padding(target,pixelated,known_arr, amount, mean_std)
    
    print(combined.shape,"\n")

    all_data = TensorDataset(combined,target,known)

    train_size = int(amount * 0.93)
    eval_size = amount - train_size
    training_data, eval_data = random_split(all_data, [train_size, eval_size])
    return training_data, eval_data