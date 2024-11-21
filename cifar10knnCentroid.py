import numpy as np # For numerical operations and array handling
import torch # PyTorch main library
import torchvision # For accessing datasets like CIFAR-10
import torchvision.transforms as transforms # For image transformations
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # The classifiers we'll use
from sklearn.metrics import accuracy_score, classification_report # For evaluation metrics
import time # For timing our training process

def load_and_prepare_data():
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL images to tensors (0-1 range)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes each channel
    ])

    # Load CIFAR-10 dataset
    # Training set
    trainset = torchvision.datasets.CIFAR10(root='./1stProject/cifar-10-batches-py', train=True,
                                          download=True, transform=transform)
    # Test set
    testset = torchvision.datasets.CIFAR10(root='./1stProject/cifar-10-batches-py', train=False,
                                         download=True, transform=transform)
    
    # Convert datasets to numpy arrays
    # This uses a DataLoader to load the entire dataset into memory as numpy arrays
    X_train = []
    y_train = []
    for images, labels in torch.utils.data.DataLoader(trainset, batch_size=len(trainset)):
        X_train = images.numpy()
        y_train = labels.numpy()

    X_test = []
    y_test = []
    for images, labels in torch.utils.data.DataLoader(testset, batch_size=len(testset)):
        X_test = images.numpy()
        y_test = labels.numpy()
    
    # Reshape the data: flatten the 32x32x3 images into 1D vectors (with 3072 features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    return X_train, X_test, y_train, y_test

def evaluate_classifier(clf, X_train, X_test, y_train, y_test, name):
    # Record start time
    start_time = time.time()
    
    # Train the classifier
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Calculate prediction accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, training_time

def main():
    # Load and prepare data
    print("Loading and preparing CIFAR-10 dataset...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Using a subset of the original data for faster execution 
    subset_size = 5000
    X_train = X_train[:subset_size]
    y_train = y_train[:subset_size]
    X_test = X_test[:1000]
    y_test = y_test[:1000]
    
    # Initialize classifiers
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn3 = KNeighborsClassifier(n_neighbors=3)
    nc = NearestCentroid()
    
    # Evaluate KNN with k=1
    print("\nEvaluating KNN (k=1)...")
    knn1_accuracy, knn1_time = evaluate_classifier(
        knn1, X_train, X_test, y_train, y_test, "KNN (k=1)"
    )
    
    # Evaluate KNN with k=3
    print("\nEvaluating KNN (k=3)...")
    knn3_accuracy, knn3_time = evaluate_classifier(
        knn3, X_train, X_test, y_train, y_test, "KNN (k=3)"
    )
    
    # Evaluate Nearest Centroid
    print("\nEvaluating Nearest Centroid...")
    nc_accuracy, nc_time = evaluate_classifier(
        nc, X_train, X_test, y_train, y_test, "Nearest Centroid"
    )
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"{'Classifier':<20} {'Accuracy':<10} {'Training Time (s)':<15}")
    print("-" * 45)
    print(f"{'KNN (k=1)':<20} {knn1_accuracy:.4f}    {knn1_time:.2f}")
    print(f"{'KNN (k=3)':<20} {knn3_accuracy:.4f}    {knn3_time:.2f}")
    print(f"{'Nearest Centroid':<20} {nc_accuracy:.4f}    {nc_time:.2f}")

if __name__ == "__main__":
    main()