import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv layer: 1 input channel, 16 output channels
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # Second conv layer: 16 input channels, 20 output channels
        self.conv2 = nn.Conv2d(16, 20, kernel_size=3, padding=1)
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(20)
        # Dropout for regularization
        self.dropout1 = nn.Dropout2d(0.2)
        self.dropout2 = nn.Dropout(0.3)
        # Fully connected layers
        self.fc1 = nn.Linear(20 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 10)
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Print parameter count during initialization
        self.print_param_count()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.dropout1(x)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = self.dropout1(x)
        x = x.view(-1, 20 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total parameters: {total_params:,}')