import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Encode class as 0 (edible) and 1 (poisonous)
    df['class'] = df['class'].map({'e': 0, 'p': 1})

    # Label encode all other categorical columns
    label_encoders = {}
    for column in df.columns[1:]:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Split into features and target
    X = df.drop(columns=['class']).values
    y = df['class'].values

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, label_encoders

class MushroomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the neural network model
class MushroomClassifier(nn.Module):
    def __init__(self, input_size):
        super(MushroomClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)  # Output layer (edible or poisonous)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluation function
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    accuracy = correct / total
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# Function to preprocess a single mushroom entry for prediction
def encode_mushroom_features(features, label_encoders):
    encoded_features = []
    for column, value in features.items():
        encoder = label_encoders.get(column)
        if encoder:
            encoded_value = encoder.transform([value])[0]
            encoded_features.append(encoded_value)
    return torch.tensor(encoded_features, dtype=torch.float32).unsqueeze(0)

# Function to predict whether the mushroom is poisonous or edible
def predict_mushroom(model, features, label_encoders):
    model.eval()
    with torch.no_grad():
        encoded_input = encode_mushroom_features(features, label_encoders)
        output = model(encoded_input)
        _, predicted = torch.max(output, 1)
        return "poisonous" if predicted.item() == 1 else "edible"

# Main function
def main(file_path):
    # Load data
    X_train, X_test, y_train, y_test, label_encoders = load_data(file_path)

    # Create datasets and data loaders
    train_dataset = MushroomDataset(X_train, y_train)
    test_dataset = MushroomDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = X_train.shape[1]
    model = MushroomClassifier(input_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=10)

    # Evaluate the model
    evaluate_model(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'mushroom_classifier.pth')
    return model, label_encoders

# Run the script
if __name__ == "__main__":
    file_path = '/home/kou/Documents/Neural-Networks/MushroomSideProject/mushrooms.csv'  # Replace with your actual file path
    model, label_encoders = main(file_path)

    # Define Mushroom 1 (poisonous) and Mushroom 2 (edible) characteristics
    mushroom_1 = {
        "cap-shape": "x", "cap-surface": "s", "cap-color": "n", "bruises": "t", 
        "odor": "p", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "n", 
        "gill-color": "k", "stalk-shape": "e", "stalk-root": "b", "stalk-surface-above-ring": "s", 
        "stalk-surface-below-ring": "s", "stalk-color-above-ring": "w", "stalk-color-below-ring": "w", 
        "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p", 
        "spore-print-color": "k", "population": "s", "habitat": "u"
    }
    mushroom_2 = {
        "cap-shape": "x", "cap-surface": "s", "cap-color": "y", "bruises": "t", 
        "odor": "a", "gill-attachment": "f", "gill-spacing": "c", "gill-size": "b", 
        "gill-color": "k", "stalk-shape": "e", "stalk-root": "c", "stalk-surface-above-ring": "s", 
        "stalk-surface-below-ring": "s", "stalk-color-above-ring": "w", "stalk-color-below-ring": "w", 
        "veil-type": "p", "veil-color": "w", "ring-number": "o", "ring-type": "p", 
        "spore-print-color": "n", "population": "n", "habitat": "g"
    }

    # Predict and print results for Mushroom 1 and Mushroom 2
    prediction_1 = predict_mushroom(model, mushroom_1, label_encoders)
    prediction_2 = predict_mushroom(model, mushroom_2, label_encoders)
    print(f"Mushroom 1 is predicted to be: {prediction_1}")
    print(f"Mushroom 2 is predicted to be: {prediction_2}")
