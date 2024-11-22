import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First conv block with moderate number of filters
        self.conv1 = nn.Conv2d(1, 48, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(48)
        
        # Second conv block with reduction
        self.conv2 = nn.Conv2d(48, 8, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(8)
        
        # Single fully connected layer
        self.fc = nn.Linear(8 * 7 * 7, 10)
        
        # Pooling and activation
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._initialize_weights()
        self.print_param_count()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = x.view(-1, 8 * 7 * 7)
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def print_param_count(self):
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Total parameters: {total_params:,}')