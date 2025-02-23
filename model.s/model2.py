import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import matplotlib.pyplot as plt
from torchvision import models
from tqdm import tqdm

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Paths
INNER_EYE_PATH = r"D:\Diversion_2k25\eye_analysis\inner_eye"
OUTER_EYE_PATH = r"D:\Diversion_2k25\eye_analysis\outer_eyes"

# Hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 10
IMAGE_SIZE = 224

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Load Datasets
inner_dataset = datasets.ImageFolder(root=INNER_EYE_PATH, transform=transform)
outer_dataset = datasets.ImageFolder(root=OUTER_EYE_PATH, transform=transform)

inner_loader = data.DataLoader(inner_dataset, batch_size=BATCH_SIZE, shuffle=True)
outer_loader = data.DataLoader(outer_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Get Class Labels
inner_classes = inner_dataset.classes
outer_classes = outer_dataset.classes

# Define CNN Model
class EyeDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(EyeDiseaseCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Train Function
def train_model(model, train_loader, num_classes, model_name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training {model_name} model...\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

    print(f"{model_name} model training complete!\n")
    return model

# Train Models
inner_eye_model = EyeDiseaseCNN(num_classes=len(inner_classes))
outer_eye_model = EyeDiseaseCNN(num_classes=len(outer_classes))

inner_eye_model = train_model(inner_eye_model, inner_loader, len(inner_classes), "Inner Eye")
outer_eye_model = train_model(outer_eye_model, outer_loader, len(outer_classes), "Outer Eye")

# Save Models
torch.save(inner_eye_model.state_dict(), "inner_eye_model.pth")
torch.save(outer_eye_model.state_dict(), "outer_eye_model.pth")

# Prediction Function
def predict(image_path, model, class_labels):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return class_labels[predicted.item()]


