import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

def training_step(network, optimizer, data, targets, bool_mask,device):
    optimizer.zero_grad()

    output = network(data).to(device=device)
    
    actual_predictions = torch.masked_select(output, bool_mask)
        
    actual_predictions = torch.clamp(actual_predictions, min=0, max=255)

    actual_targets = torch.masked_select(targets, bool_mask).float()

    loss = F.mse_loss(actual_predictions, actual_targets)
    loss.backward()
    optimizer.step()
    return torch.sqrt(loss).item()


def eval_step(network, data, targets, bool_mask,device,epoch,todo_delete):
    with torch.no_grad():
        output = network(data).to(device=device)
        actual_predictions = torch.masked_select(output, bool_mask)
        actual_predictions = torch.clamp(actual_predictions, min=0, max=255)
        actual_targets = torch.masked_select(targets, bool_mask).float()
        loss = F.mse_loss(actual_predictions, actual_targets)
        return torch.sqrt(loss).item()


def training_loop(
        network: torch.nn.Module,
        train_data: torch.utils.data.Dataset,
        eval_data: torch.utils.data.Dataset,
        num_epochs: int,
        mean_std: tuple,
        show_progress: bool = True) -> tuple[list, list]:
    
    device = "cuda"
    device = torch.device(device)
    if not torch.cuda.is_available():
        print("CUDA IS NOT AVAILABLE")
        device = torch.device("cpu")
    
    batch_size = 32
    optimizer = torch.optim.Adam(network.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=batch_size, shuffle=False)
    train_losses = []
    eval_losses = []
    for epoch in tqdm(range(num_epochs), desc="Epoch", position=0, disable=not show_progress):

        network.train()
        epoch_train_losses = []

        for data, targets, bool_mask in tqdm(train_loader, desc="Minibatch", position=1, leave=False, disable=not show_progress):
            data = data.to(device=device)
            targets = targets.to(device=device)
            bool_mask = bool_mask.to(device=device)
            loss = training_step(network, optimizer, data, targets,bool_mask,device)
            epoch_train_losses.append(loss)
        train_losses.append(torch.mean(torch.tensor(epoch_train_losses)))
        
        network.eval()

        epoch_eval_losses = []
        todo_delete = 0
        for data, targets, bool_mask in eval_loader:
            data = data.to(device=device)
            targets = targets.to(device=device)
            bool_mask = bool_mask.to(device=device)
            loss = eval_step(network, data, targets, bool_mask,device,epoch,todo_delete)
            todo_delete = 1
            epoch_eval_losses.append(loss)
        eval_losses.append(torch.mean(torch.tensor(epoch_eval_losses)))

        scheduler.step()
        print('\n',torch.mean(torch.tensor(epoch_train_losses)),"\t<- Train|Eval ->",torch.mean(torch.tensor(epoch_eval_losses)),'\n')
        # Early stopping
        if np.argmin(eval_losses) <= epoch - 4:
            break
            
    return network, train_losses, eval_losses


def plot_losses(train_losses: list, eval_losses: list):
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train loss (rmse)")
    ax.plot(eval_losses, label="Eval loss (rmse)")
    ax.legend()
    ax.set_xlim(0, len(train_losses) - 1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.show()
    plt.close(fig)