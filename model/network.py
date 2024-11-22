import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv layer: 1 input channel, 8 output channels
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        # Second conv layer: 8 input channels, 16 output channels
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Reduced from 128 to 64
        self.fc2 = nn.Linear(64, 10)
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Print parameter count during initialization
        self.print_param_count()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 28x28 -> 14x14
        x = self.dropout1(x)
        x = self.pool(self.relu(self.conv2(x)))  # 14x14 -> 7x7
        x = self.dropout1(x)
        x = x.view(-1, 16 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total parameters: {total_params:,}')