# CIFAR-10 Image Classification using Convolutional Neural Networks

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import time

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data Preprocessing
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./1stProject/cifar-10-batches-py', train=True,
                                      download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128,
                        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./1stProject/cifar-10-batches-py', train=False,
                                     download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100,
                       shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Visualization function
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('./1stProject/cifar10.png')

# Get random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Show images
print('Some example images:')
imshow(torchvision.utils.make_grid(images[:4]))
print('Labels:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
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

# Initialize the model
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Training function
def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 100 == 99:
            print(f'[{i + 1}] loss: {running_loss / 100:.3f} | acc: {100.*correct/total:.2f}%')
            running_loss = 0.0
    
    return 100. * correct / total

# Evaluation function
def evaluate(model, testloader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(testloader)
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy, avg_loss

# Training loop
num_epochs = 50
best_acc = 0
train_acc_history = []
test_acc_history = []

print(f"Training on {device}")
start_time = time.time()

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}')
    
    train_acc = train(model, trainloader, criterion, optimizer, device)
    test_acc, test_loss = evaluate(model, testloader, criterion, device)
    
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)
    
    # Learning rate scheduling
    scheduler.step(test_loss)
    
    # Save best model
    if test_acc > best_acc:
        print(f'Saving best model with accuracy: {test_acc:.2f}%')
        torch.save(model.state_dict(), './1stProject/cifar10_best.pth')
        best_acc = test_acc

training_time = time.time() - start_time
print(f'\nTraining completed in {training_time/60:.2f} minutes')
print(f'Best accuracy: {best_acc:.2f}%')

# Plot training history
plt.figure(figsize=(10, 5))
plt.plot(train_acc_history, label='Train Accuracy')
plt.plot(test_acc_history, label='Test Accuracy')
plt.title('Model Accuracy over Time')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('./1stProject/cifar10_accuracy.png')

# Load best model and evaluate on test set
model.load_state_dict(torch.load('./1stProject/cifar10_best.pth'))
test_acc, _ = evaluate(model, testloader, criterion, device)

# Visualization of predictions
model.eval()
dataiter = iter(testloader)
images, labels = next(dataiter)
images = images.to(device)
labels = labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Show some test images and their predictions
print('\nSome test predictions:')
imshow(torchvision.utils.make_grid(images[:4].cpu()))
print('Ground Truth:', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
print('Predicted:', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

# Compute per-class accuracy
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

print('\nPer-class accuracy:')
for i in range(10):
    print(f'{classes[i]:>5s}: {100 * class_correct[i] / class_total[i]:.2f}%')

# Output:
# Training completed in 607.96 minutes
# Best accuracy: 91.27%
# /home/kou/Documents/Neural-Networks/1stProject/cipher10.py:206: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
#   plt.show()
# /home/kou/Documents/Neural-Networks/1stProject/cipher10.py:209: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   model.load_state_dict(torch.load('cifar10_best.pth'))
# Test set: Average loss: 0.3188, Accuracy: 91.27%

# Some test predictions:
# Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). Got range [-0.71453285..1.8476002].
# Ground Truth:   cat  ship  ship plane
# Predicted:   cat  ship  ship plane

# Per-class accuracy:
# plane: 97.73%
#   car: 100.00%
#  bird: 78.95%
#   cat: 76.74%
#  deer: 91.84%
#   dog: 87.50%
#  frog: 94.34%
# horse: 93.75%
#  ship: 97.92%
# truck: 96.55%