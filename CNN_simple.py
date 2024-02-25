import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            # num_hidden_layers: int,
            num_classes: int,
            use_batch_normalization: bool = True,
            kernel_size: int = 3,
            activation_function: nn.Module = nn.ReLU()): #ELU
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.num_hidden_layers = len(hidden_channels)
        self.use_batch_normalization = use_batch_normalization
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.activation_function = activation_function

        self.conv_layers = nn.ModuleList()
        self.dropout = torch.nn.Dropout(0.15)
        
        self.conv_layers.append(nn.Conv2d(
            input_channels,
            hidden_channels[0],
            3,
            padding="same",
            padding_mode="zeros"
        ))

        for num in range(1, self.num_hidden_layers):
            self.conv_layers.append(nn.Conv2d(
                hidden_channels[num-1],
                hidden_channels[num],
                kernel_size,
                padding="same",
                padding_mode="zeros"
            ))
        if self.use_batch_normalization:
            self.batch_norm_layers = nn.ModuleList()
            for i in range(self.num_hidden_layers):
                self.batch_norm_layers.append(nn.BatchNorm2d(hidden_channels[i]))
        self.output_layer = nn.Conv2d(hidden_channels[-1], self.num_classes, kernel_size=1, stride=1)
    
    
    def forward(self, input_images: torch.Tensor) -> torch.Tensor:

        for i in range(self.num_hidden_layers):
            input_images = self.conv_layers[i](input_images)

            if self.use_batch_normalization:
                input_images = self.batch_norm_layers[i](input_images)
            input_images = self.activation_function(input_images)
        input_images = self.output_layer(input_images)


        return input_images
