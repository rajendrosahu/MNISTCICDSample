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
    
    # Use the same normalization as in training
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load test dataset
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
    assert accuracy >= 93, f"Model accuracy is {accuracy:.2f}%, should be >= 93%"

def test_model_output_range():
    """Test if model outputs are proper logits (before softmax)"""
    model = SimpleCNN()
    model.eval()
    
    # Use normalized input like in training
    test_input = torch.randn(10, 1, 28, 28)
    test_input = (test_input - 0.1307) / 0.3081
    
    with torch.no_grad():
        outputs = model(test_input)
    
    # Check if outputs are in a reasonable range for logits
    assert torch.max(outputs) < 150, "Logits too large"
    assert torch.min(outputs) > -150, "Logits too negative"
    assert not torch.isnan(outputs).any(), "Model produced NaN values"

def test_batch_invariance():
    """Test if model produces same output regardless of batch size"""
    model = SimpleCNN()
    model.eval()
    
    # Use normalized input like in training
    test_input = torch.randn(1, 1, 28, 28)
    test_input = (test_input - 0.1307) / 0.3081
    
    with torch.no_grad():
        single_output = model(test_input)
        batch_input = test_input.repeat(5, 1, 1, 1)
        batch_output = model(batch_input)
    
    # Check if all outputs in batch are similar (not exactly identical due to batch norm)
    for i in range(5):
        assert torch.allclose(single_output, batch_output[i], rtol=1e-3, atol=1e-3), \
            "Model produces too different outputs for same input in different batch sizes"

def test_intermediate_activations():
    """Test if intermediate feature maps have reasonable values"""
    model = SimpleCNN()
    model.eval()
    
    # Use normalized input like in training
    test_input = torch.randn(1, 1, 28, 28)
    test_input = (test_input - 0.1307) / 0.3081
    
    # Register hooks to capture intermediate activations
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output
        return hook
    
    # Register hooks after ReLU layers
    model.relu.register_forward_hook(get_activation('relu1'))
    
    # Forward pass
    with torch.no_grad():
        model(test_input)
    
    # Check activations
    for name, activation in activations.items():
        assert torch.all(activation >= 0), f"Found negative values after ReLU in {name}"
        assert not torch.isnan(activation).any(), f"Found NaN values in {name}"