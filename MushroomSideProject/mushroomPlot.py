import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the mushroom data from CSV file
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Initialize label encoder
    le = LabelEncoder()
    
    # Convert all categorical columns to numerical values
    encoded_df = df.copy()
    for column in df.columns:
        encoded_df[column] = le.fit_transform(df[column])
    
    return encoded_df

def calculate_distance(row1, row2):
    """
    Calculate Euclidean distance between two rows
    """
    return np.sqrt(sum((row1 - row2) ** 2))

def find_nearest_neighbors(sample, data, k=3):
    """
    Find k nearest neighbors for a given sample
    """
    distances = []
    
    # Calculate distance from sample to each row in dataset
    for idx, row in data.iterrows():
        dist = calculate_distance(sample, row)
        distances.append((dist, idx))
    
    # Sort distances and get k nearest neighbors
    distances.sort()
    neighbors = distances[1:k+1]  # Skip first if sample is from dataset
    
    return neighbors

def calculate_centroids(data):
    """
    Calculate centroids for each class (edible and poisonous)
    """
    # Separate data by class
    class_column = data.columns[0]  # Assuming first column is class
    edible = data[data[class_column] == 0]
    poisonous = data[data[class_column] == 1]
    
    # Calculate centroids
    edible_centroid = edible.mean()
    poisonous_centroid = poisonous.mean()
    
    return edible_centroid, poisonous_centroid

def visualize_mushrooms(data, random_mushroom=None, neighbors=None):
    """
    Visualize mushroom data points and centroids using PCA
    """
    # Separate features and labels
    X = data.iloc[:, :].values
    y = data.iloc[:, 0].values  # First column is class
    
    # Apply PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot all points
    edible_mask = y == 0
    plt.scatter(X_pca[edible_mask, 0], X_pca[edible_mask, 1], 
               c='green', label='Edible', alpha=0.5, s=50)
    plt.scatter(X_pca[~edible_mask, 0], X_pca[~edible_mask, 1], 
               c='red', label='Poisonous', alpha=0.5, s=50)
    
    # Calculate and plot centroids in PCA space
    edible_centroid = np.mean(X_pca[edible_mask], axis=0)
    poisonous_centroid = np.mean(X_pca[~edible_mask], axis=0)
    
    plt.scatter(edible_centroid[0], edible_centroid[1], 
               c='darkgreen', marker='*', s=300, label='Edible Centroid')
    plt.scatter(poisonous_centroid[0], poisonous_centroid[1], 
               c='darkred', marker='*', s=300, label='Poisonous Centroid')
    
    # If random mushroom is provided, plot it and its neighbors
    if random_mushroom is not None:
        random_mushroom_pca = pca.transform([random_mushroom.values])[0]
        plt.scatter(random_mushroom_pca[0], random_mushroom_pca[1], 
                   c='blue', marker='D', s=100, label='Random Mushroom')
        
        if neighbors is not None:
            for _, idx in neighbors:
                neighbor_pca = pca.transform([data.iloc[idx].values])[0]
                plt.scatter(neighbor_pca[0], neighbor_pca[1], 
                          c='cyan', marker='s', s=100)
            plt.scatter([], [], c='cyan', marker='s', s=100, 
                       label='Nearest Neighbors')
    
    plt.title('Mushroom Dataset Visualization (PCA)')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output.png')

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('/home/kou/Documents/Neural-Networks/MushroomSideProject/mushrooms.csv')
    
    # Select random mushroom
    random_idx = random.randint(0, len(data)-1)
    random_mushroom = data.iloc[random_idx]
    
    print("Random Mushroom Selected:")
    print(f"Index: {random_idx}")
    print(f"Class: {'Edible' if random_mushroom.iloc[0] == 0 else 'Poisonous'}")
    
    # Find 3 nearest neighbors
    neighbors = find_nearest_neighbors(random_mushroom, data)
    
    # Calculate centroids
    edible_centroid, poisonous_centroid = calculate_centroids(data)
    
    # Visualize the data
    visualize_mushrooms(data, random_mushroom, neighbors)

if __name__ == "__main__":
    main()