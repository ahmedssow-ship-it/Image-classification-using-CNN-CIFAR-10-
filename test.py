
    # Imports
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# PyTorch libraries
import torch
from torch import nn
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
#from model import CNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('baseline_model.pth'))
model.eval()   # mode évaluation


# Transformations
transform = transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomCrop(32, padding=4),
  transforms.ToTensor(),
  transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Load the testing samples
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transform
  )
testloader = torch.utils.data.DataLoader(test_data, batch_size=64,
                                        shuffle=False)

test_loss = 0.0
all_preds = []
all_labels = []

with torch.no_grad():  # désactive les gradients
  for inputs, labels in testloader:
      inputs, labels = inputs.to(device), labels.to(device)

      outputs = model(inputs)
      loss = criterion(outputs, labels)
      test_loss += loss.item()

      # Prédictions
      _, predicted = torch.max(outputs, 1)

      all_preds.extend(predicted.cpu().numpy())
      all_labels.extend(labels.cpu().numpy())

# Calcul des métriques
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')

print("
===== TEST RESULTS =====")
print(f"Test Loss: {test_loss/len(testloader):.4f}")
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
    