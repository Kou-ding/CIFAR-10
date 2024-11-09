import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder
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
    edible = data[data[class_column] == 0]  # Assuming 0 = edible, 1 = poisonous
    poisonous = data[data[class_column] == 1]
    
    # Calculate centroids
    edible_centroid = edible.mean()
    poisonous_centroid = poisonous.mean()
    
    return edible_centroid, poisonous_centroid

def main():
    # Load and preprocess data
    data = load_and_preprocess_data('/home/kou/Documents/Neural-Networks/MushroomSideProject/mushrooms.csv')
    
    # Select random mushroom
    random_idx = random.randint(0, len(data)-1)
    random_mushroom = data.iloc[random_idx]
    
    print("Random Mushroom Selected:")
    print(f"Index: {random_idx}")
    # Fixed the deprecation warning by using iloc to access the class column
    print(f"Class: {'Edible' if random_mushroom.iloc[0] == 0 else 'Poisonous'}")
    print("\nFeatures:")
    for col, val in random_mushroom.items():
        print(f"{col}: {val}")
    
    # Find 3 nearest neighbors
    neighbors = find_nearest_neighbors(random_mushroom, data)
    
    print("\n3 Nearest Neighbors:")
    for dist, idx in neighbors:
        neighbor = data.iloc[idx]
        # Fixed here as well
        print(f"\nNeighbor at index {idx}")
        print(f"Distance: {dist:.4f}")
        print(f"Class: {'Edible' if neighbor.iloc[0] == 0 else 'Poisonous'}")
    
    # Calculate centroids
    edible_centroid, poisonous_centroid = calculate_centroids(data)
    
    print("\nCentroids:")
    print("\nEdible Mushroom Centroid:")
    for col, val in edible_centroid.items():
        print(f"{col}: {val:.4f}")
    
    print("\nPoisonous Mushroom Centroid:")
    for col, val in poisonous_centroid.items():
        print(f"{col}: {val:.4f}")
    
    # Find closest centroid to random mushroom
    dist_to_edible = calculate_distance(random_mushroom, edible_centroid)
    dist_to_poisonous = calculate_distance(random_mushroom, poisonous_centroid)
    
    print("\nDistance to Centroids:")
    print(f"Distance to Edible Centroid: {dist_to_edible:.4f}")
    print(f"Distance to Poisonous Centroid: {dist_to_poisonous:.4f}")
    print(f"Closest Centroid: {'Edible' if dist_to_edible < dist_to_poisonous else 'Poisonous'}")

if __name__ == "__main__":
    main()