from prepare_images import Prepare_Images
from dataset import get_dataset
from CNN_simple import SimpleCNN
from padding import stack_with_padding, stack_with_padding_just_for_test
from training_loop import training_loop, plot_losses
import torch
import pickle as pkl
import numpy as np
from submission_serialization import serialize
from tqdm import tqdm
from PIL import Image

print(torch.cuda.get_device_name(0))

#Test sets of images
Images = Prepare_Images(r"train") # 35 000 samples
print("Test1 created\n")

target_image,pixelated_image,known_arr, mean_std = Images.prepare_images()
print("Images are ready\n")

amount = len(Images)
print(f"Length was calculated and is: {amount}\n")

train_data, eval_data = get_dataset(target_image,pixelated_image,known_arr,amount, mean_std)
print("Dataset is ready\n")

device = torch.device("cuda")
if not torch.cuda.is_available():
    print("CUDA IS NOT AVAILABLE")
    device = torch.device("cpu")



network = SimpleCNN(2, [32,64,128,128,256,256,128,128,64,32], 1, False, 3).to(device)

print("Network was created\n")

params = sum(param.numel() for param in network.parameters())
print(network)
print(params)

network_final, train_losses, eval_losses = training_loop(network, train_data, eval_data,33,mean_std)

print(f"Training loop was done and the results are:\n{train_losses, eval_losses}\n")

with open(r"test_set.pkl", "rb") as f:
    data = pkl.load(f)
    pixelated_image_test_raw = data["pixelated_images"]
    known_arr_test_raw = data["known_arrays"]

combined_arr_test,known_arr_test = stack_with_padding_just_for_test(pixelated_image_test_raw,known_arr_test_raw,len(pixelated_image_test_raw))

predictions_list = []
print(f"All in all items to check: {len(combined_arr_test)}")
def step_(combined,known,device,iter):
    with torch.no_grad():
        combined = combined.to(device=device)
        known = known.to(device=device)
        prediction = network_final(combined.view(1,2,64,64))

        actual_prediction = (torch.masked_select(prediction, known)).to(device=device)
        actual_prediction = torch.clamp(actual_prediction, min=0, max=255)
        prediction_mod = actual_prediction.detach().cpu().numpy().astype(dtype=np.uint8)
        
        predictions_list.append(prediction_mod.flatten())

i = 0
for combined, known in tqdm(zip(combined_arr_test,known_arr_test)):
    t = step_(combined,known,device,i)
    i = i + 1
    if t == 2:
        break

serial = serialize(predictions_list, r"prediction_0.txt")



plot_losses(train_losses, eval_losses)
print("Plot was drawn, It is done, Have a great day\n")