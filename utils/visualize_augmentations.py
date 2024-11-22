import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import numpy as np

def save_augmented_examples():
    # Create directory for images if it doesn't exist
    os.makedirs('docs/images', exist_ok=True)
    
    # Define the augmentation pipeline
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
        ),
        transforms.ToTensor(),
    ])
    
    # Load a single MNIST image
    dataset = datasets.MNIST('data', train=True, download=True)
    
    # Create a figure with original and augmented images
    plt.figure(figsize=(15, 3))
    
    # Convert PIL image to numpy array for original image
    original_image = np.array(dataset[0][0])
    
    # Show original image
    plt.subplot(1, 5, 1)
    plt.title('Original')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    # Show 4 different augmentations
    for i in range(4):
        # Convert to PIL Image first
        pil_image = transforms.ToPILImage()(torch.tensor(original_image).unsqueeze(0).float())
        # Apply augmentation
        augmented = transform(pil_image)
        
        plt.subplot(1, 5, i+2)
        plt.title(f'Augmented {i+1}')
        plt.imshow(augmented.squeeze(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('docs/images/augmentations.png')
    plt.close()

if __name__ == "__main__":
    save_augmented_examples() 