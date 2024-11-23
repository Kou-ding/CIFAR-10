import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import random

# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer 
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Second convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Third convolutional layer
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

def load_model(model_path):
    """Load the trained model from .pth file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model, device

def prepare_image(image_path, size=(32, 32)):
    """Prepare a custom image for inference using the same transformations as training"""

    # Open the image
    image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
    
    # Apply transformations to get the tensor
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    image_tensor = transform(image).unsqueeze(0)
    
    # Denormalize for visualization
    denormalize = transforms.Normalize(
        mean=[-0.4914 / 0.2023, -0.4822 / 0.1994, -0.4465 / 0.2010],
        std=[1 / 0.2023, 1 / 0.1994, 1 / 0.2010]
    )
    denormalized_tensor = denormalize(image_tensor.squeeze(0))
    resized_image = denormalized_tensor.permute(1, 2, 0).clamp(0, 1).numpy()  # Convert to NumPy and clamp values to [0, 1]
    resized_image = (resized_image * 255).astype(np.uint8)  # Scale to [0, 255]
    
    # Debug information
    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Value range: [{image_tensor.min():.2f}, {image_tensor.max():.2f}]")
    
    return image_tensor, resized_image

def get_random_cifar_image():
    """Get a random image from CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py', train=False,
                                          download=True, transform=transform)
    idx = random.randint(0, len(testset) - 1)
    image, label = testset[idx]
    return image.unsqueeze(0), label, testset.data[idx]

def predict_image(model, image_tensor, device):
    """Make a prediction on the input image"""
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        return predicted.item(), probs[0].cpu()

def display_prediction(image, prediction, probabilities=None, title=None):
    """Display the image and prediction"""
    plt.figure(figsize=(6, 6))
    
    # If image is a PIL Image, convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    # If image is a tensor, convert to numpy and denormalize
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((1, 2, 0))
        # Denormalize
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2023, 0.1994, 0.2010])
        image = std * image + mean
    # If image is uint8 numpy array
    elif isinstance(image, np.ndarray) and image.dtype == np.uint8:
        image = image / 255.0
    
    # Ensure image is in correct range
    plt.imshow(np.clip(image, 0, 1))
    
    if probabilities is not None:
        pred_text = f'Prediction: {classes[prediction]} ({probabilities[prediction]*100:.2f}%)'
    else:
        pred_text = f'Prediction: {classes[prediction]}'
    
    if title:
        plt.title(f'{title}\n{pred_text}')
    else:
        plt.title(pred_text)
    plt.axis('off')
    plt.show()

def main():
    # Model path
    model_path = './cifar10_best.pth'
    
    try:
        model, device = load_model(model_path)
        print(f"Model loaded successfully. Running on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    while True:
        print("\nChoose an option:")
        print("1. Test random CIFAR-10 image")
        print("2. Test custom image")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            image_tensor, true_label, original_image = get_random_cifar_image()
            pred, probs = predict_image(model, image_tensor, device)
            display_prediction(original_image/255.0, pred, probs, 
                             f'True Label: {classes[true_label]}')
            
        elif choice == '2':
            image_name = input("Enter the image's name: ")
            image_path = './cifar10tests/' + image_name
            try:
                print("\nProcessing image...")
                image_tensor, original_image = prepare_image(image_path)
                print("Making prediction...")
                pred, probs = predict_image(model, image_tensor, device)
                display_prediction(original_image, pred, probs)
                
                # Print top-3 predictions
                top_probs, top_labels = torch.topk(probs, 3)
                print("\nTop 3 predictions:")
                for i in range(3):
                    print(f"{classes[top_labels[i]]}: {top_probs[i]*100:.2f}%")
                    
            except Exception as e:
                print(f"Error processing image: {e}")
                
        elif choice == '3':
            print("Exiting...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()