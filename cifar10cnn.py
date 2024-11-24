import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

# Specify the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

########################## Useful Functions ########################### 
# Training function
def train(model, trainloader, criterion, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0 #
    correct = 0 # Number of correctly predicted images
    total = 0 # Total number of images
    
    # Iterate over the training dataset
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) #data[0] is the image, data[1] is the label
        
        optimizer.zero_grad() # Reset the gradients of model parameters
        outputs = model(inputs) # Forward pass
        loss = criterion(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        # Track training statistics
        running_loss += loss.item() # Add loss to running loss
        _, predicted = outputs.max(1) # Get the class index with the highest probability
        total += labels.size(0) # Add batch size to total
        correct += predicted.eq(labels).sum().item() # Add number of correct predictions to correct
        
        # Print statistics every 100 mini-batches
        if i % 100 == 99:
            print(f'[{i + 1}] loss: {running_loss / 100:.3f} | acc: {100.*correct/total:.2f}%')
            running_loss = 0.0 # Reset running loss
    
    return 100. * correct / total

# Evaluation function
def evaluate(model, testloader, criterion, device):
    model.eval() # Set model to evaluation mode
    test_loss = 0 # Total loss
    correct = 0 # Number of correctly predicted images
    total = 0 # Total number of images
    
    # Compute per-class accuracy
    class_correct = [0]*10
    class_total = [0]*10

    with torch.no_grad(): # Disable gradient calculation
        # Iterate over the test dataset
        for data in testloader: 
            images, labels = data[0].to(device), data[1].to(device) # data[0] is the image, data[1] is the label
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # Calculate loss
            
            # Track test statistics
            test_loss += loss.item() # Add loss to test loss
            _, predicted = outputs.max(1) # Get the class index with the highest probability
            total += labels.size(0) # Add batch size to total
            correct += predicted.eq(labels).sum().item() # Add number of correct predictions to correct

            # Update per-class statistics
            for i in range(len(labels)):  # Loop through each image in the batch
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()  # Increment correct count for this class
                class_total[label] += 1  # Increment total count for this class

    # Calculate overall test accuracy and average loss            
    accuracy = 100. * correct / total # Calculate accuracy
    avg_loss = test_loss / len(testloader) # Calculate average loss
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Calculate per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0: # Avoid division by zero
            class_accuracy = 100. * class_correct[i] / class_total[i]
            print(f'{classes[i]:>5s}: {class_accuracy:.2f}%')

    return accuracy, avg_loss

####################### Define the CNN architecture #######################
class CNN(nn.Module):
    # Initialize the architecture
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 32, 3, padding=1), # 3 input channels (R G B), 32 output channels, 3x3 kernel, 1 padding
            nn.BatchNorm2d(32), # normalize the output of the previous layer to have a mean = 0 and standard deviation = 1
            nn.ReLU(), # ReLU activation function: f(x) = max(0, x)

            # Second convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  # for each 2x2 window, take the maximum value
            nn.Dropout(0.25), # set 25% of the neurons to zero

            # Third convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Fourth convolutional layer
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Fifth convolutional layer
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Sixth convolutional layer
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512), # First fully connected layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10) # Final fully connected layer, 512 input features, 10 output features (classes)
        )

    # Define the forward pass
    def forward(self, x):
        x = self.conv_layers(x) # Pass input through convolutional layers
        x = x.view(x.size(0), -1) # Flatten the output of the convolutional layers
        x = self.fc_layers(x) # Pass through fully connected layers
        return x
############################## Main Function ###############################
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # The variable "device" specifies the computational device 
    # This is where we run our neural network on (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################### Data Preprocessing #########################
    # Tranformations for the training dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Randomly crops image with padding
        transforms.RandomHorizontalFlip(), # Randomly flips image horizontally
        transforms.ToTensor(), # Converts image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # Tranformations for the test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(), # Converts image to tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    ############################# Our Dataset #############################
    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py', train=True,
                                        download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128,
                            shuffle=True, num_workers=2)
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./cifar-10-batches-py', train=False,
                                        download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100,
                        shuffle=False, num_workers=2)
        
    ######################### Model Initialization ##########################
    model = CNN().to(device) # Send model to device for training
    criterion = nn.CrossEntropyLoss() # Loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Optimizer
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) # Learning rate scheduler

    # Main loop
    num_epochs = 50 # Number of epochs
    best_acc = 0 # Best accuracy
    train_acc_history = [] # Training accuracy history
    test_acc_history = [] # Test accuracy history

    print(f"Training on {device}") # Print device
    start_time = time.time() # Start time

    # Iterate over the epochs
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}') # Print epoch
        
        # Train and evaluate the model
        train_acc = train(model, trainloader, criterion, optimizer, device)
        test_acc, test_loss = evaluate(model, testloader, criterion, device)
        
        # Track accuracy history
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss) # Adjust learning rate based on loss
        
        # Save best model
        if test_acc > best_acc:
            print(f'Saving best model with accuracy: {test_acc:.2f}%') # Print current best accuracy
            # Save the model with the best accuracy in the filename
            torch.save(model.state_dict(), f'./cifar10_acc_{test_acc:.2f}.pth')
            best_acc = test_acc

    # Calculate training time
    training_time = time.time() - start_time # Calculate training time
    print(f'\nTraining completed in {training_time/60:.2f} minutes') # Print training time
    print(f'Best accuracy: {best_acc:.2f}%') # Print final best accuracy

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.title('Model Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('cifar10_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()