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
    
    # Load MNIST dataset with minimal augmentation
    transform_train = transforms.Compose([
        transforms.RandomRotation(2),  # Very minimal rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.02, 0.02),  # Very minimal translation
            scale=(0.98, 1.02),  # Very minimal scaling
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)  # Smaller batch size
    
    # Initialize model
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Use SGD with momentum
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,  # Moderate learning rate
        momentum=0.9,
        nesterov=True
    )
    
    # Custom learning rate schedule
    def adjust_learning_rate(optimizer, progress):
        if progress < 0.1:  # First 10% - warm up
            lr = 0.05 * (progress / 0.1)
        elif progress < 0.8:  # Next 70% - constant high learning rate
            lr = 0.05
        else:  # Last 20% - cosine decay
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor((progress - 0.8) / 0.2 * 3.14159)))
            lr = 0.05 * cosine_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    # Train for 1 epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    best_acc = 0.0
    best_model_state = None
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        # Update learning rate
        progress = batch_idx / len(train_loader)
        current_lr = adjust_learning_rate(optimizer, progress)
        
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        running_loss += loss.item()
        
        # Calculate current accuracy
        acc = 100 * correct / total
        
        # Save best model
        if acc > best_acc:
            best_acc = acc
            best_model_state = model.state_dict().copy()
        
        # Update progress bar
        if batch_idx % 5 == 0:
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.3f}',
                'acc': f'{acc:.2f}%',
                'best_acc': f'{best_acc:.2f}%',
                'lr': f'{current_lr:.4f}'
            })
    
    final_acc = 100 * correct / total
    print(f'\nTraining accuracy: {final_acc:.2f}%')
    print(f'Best accuracy seen: {best_acc:.2f}%')
    
    # Always use the best model state
    if best_acc > final_acc:
        print(f"Using best model checkpoint with accuracy: {best_acc:.2f}%")
        model.load_state_dict(best_model_state)
        final_acc = best_acc
    
    # Save model with timestamp and accuracy
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not os.path.exists('models'):
        os.makedirs('models')
    model_path = f'models/model_{timestamp}_acc{final_acc:.1f}.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

if __name__ == "__main__":
    train() 