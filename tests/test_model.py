import torch
import pytest
from model.network import SimpleCNN
from torchvision import datasets, transforms
from tqdm import tqdm

def test_model_parameters():
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25,000"
    print(f"Model has {total_params} parameters")

def test_input_output_dimensions():
    model = SimpleCNN()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, should be (1, 10)"

def test_model_accuracy():
    device = torch.device("cpu")
    model = SimpleCNN().to(device)
    
    # Load the latest model
    import glob
    import os
    model_files = glob.glob('models/*.pth')
    assert len(model_files) > 0, "No trained model found in models directory"
    latest_model = max(model_files, key=os.path.getctime)
    model.load_state_dict(torch.load(latest_model, map_location=device))
    
    # Load test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    
    model.eval()
    correct = 0
    total = 0
    
    print("\nTesting model accuracy...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nTest accuracy: {accuracy:.2f}%")
    assert accuracy >= 95, f"Model accuracy is {accuracy:.2f}%, should be >= 95%" 