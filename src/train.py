import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import ssl
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import ReseauMNIST

# --- Contournement SSL ---
ssl._create_default_https_context = ssl._create_unverified_context

# --- Paramètres ---
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)

# --- Données ---
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = torchvision.datasets.MNIST(
    DATA_PATH, train=True, download=False, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    DATA_PATH, train=False, download=False, transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Modèle, loss, optimiseur ---
model = ReseauMNIST()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Boucle d'entraînement ---
for epoch in range(EPOCHS):

    # Phase entraînement
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Phase évaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(
        f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Accuracy: {accuracy:.2f}%"
    )

# --- Sauvegarde ---
torch.save(model.state_dict(), "model_mnist.pth")
print("Modèle sauvegardé !")
