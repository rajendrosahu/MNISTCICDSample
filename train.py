import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model.network import SimpleCNN
from datetime import datetime
import os
from tqdm import tqdm

def train():
    # Force CPU usage
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset with augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.002, 
                                            steps_per_epoch=len(train_loader), 
                                            epochs=1)
    
    # Train for 1 epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Update progress bar
        if batch_idx % 10 == 0:
            acc = 100 * correct / total
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{acc:.2f}%'
            })
    
    final_acc = 100 * correct / total
    print(f'\nTraining accuracy: {final_acc:.2f}%')
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = f'models/model_{timestamp}_acc{final_acc:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")
    
if __name__ == "__main__":
    train() 