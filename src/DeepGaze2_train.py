import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataset import GazeDataset

import matplotlib.pyplot as plt
import seaborn as sns


def plot_ground_truth(dataset):
    x_coords = []
    y_coords = []

    for entry in dataset.data:
        img_path, x, y = entry  # Korrekte Extraktion der Werte
        x_coords.append(x)
        y_coords.append(y)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(x=x_coords, y=y_coords, cmap='Blues', fill=True)
    plt.scatter(x_coords, y_coords, s=5, color='red', alpha=0.5)
    plt.xlim(0, 320)
    plt.ylim(0, 160)
    plt.title("Verteilung der Ground-Truth-Blickpunkte")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.show()


# Bestehende Funktionen
def plot_ground_truth(dataset):
    x_coords = []
    y_coords = []

    for entry in dataset.data:
        img_path, x, y = entry  # Korrekte Extraktion der Werte
        x_coords.append(x)
        y_coords.append(y)

    plt.figure(figsize=(8, 5))
    sns.kdeplot(x=x_coords, y=y_coords, cmap='Blues', fill=True)
    plt.scatter(x_coords, y_coords, s=5, color='red', alpha=0.5)
    plt.xlim(0, 320)
    plt.ylim(0, 160)
    plt.title("Verteilung der Ground-Truth-Blickpunkte")
    plt.xlabel("X-Koordinate")
    plt.ylabel("Y-Koordinate")
    plt.show()


import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_random_sample(dataset, model):
    """
    Wählt ein zufälliges Bild aus dem Dataset aus, lädt die dazugehörigen Ground-Truth-Blickpunkte,
    skaliert sie korrekt und vergleicht sie mit der Vorhersage des Modells.
    """

    # Zufälliges Sample auswählen
    img_path, x_gt, y_gt = random.choice(dataset.data)

    # Bild laden
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # **Ground-Truth-Koordinaten korrekt skalieren**
    original_width, original_height = 1920, 1080  # Ursprüngliche Größe
    heatmap_width, heatmap_height = 320, 160  # Heatmap-Größe

    x_gt_scaled = int(x_gt * (heatmap_width / original_width))
    y_gt_scaled = int(y_gt * (heatmap_height / original_height))

    # **Ground-Truth-Heatmap erzeugen**
    heatmap_gt = np.zeros((heatmap_height, heatmap_width))
    heatmap_gt[y_gt_scaled, x_gt_scaled] = 1  # Ground-Truth-Position setzen

    # **Modell-Vorhersage**
    image_tensor = dataset.transform(image).unsqueeze(0)  # Falls nötig, transformieren
    with torch.no_grad():
        predicted_heatmap = model(image_tensor)  # Modell-Vorhersage

    # **Visualisierung**
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Originalbild mit Ground-Truth
    axes[0].imshow(image)
    axes[0].scatter(x_gt, y_gt, color='red', s=50, label="Ground Truth")
    axes[0].set_title("Originalbild mit Ground-Truth-Punkt")
    axes[0].legend()

    # Ground-Truth-Heatmap
    axes[1].imshow(image, alpha=0.5)
    axes[1].imshow(heatmap_gt, cmap="jet", alpha=0.6)
    axes[1].set_title("Ground-Truth-Heatmap")

    # Modell-Vorhersage-Heatmap
    axes[2].imshow(image, alpha=0.5)
    axes[2].imshow(predicted_heatmap.squeeze(), cmap="jet", alpha=0.6)
    axes[2].set_title("Modell-Vorhersage-Heatmap")

    plt.show()

# Beispiel-Aufruf:
# plot_random_sample(dataset, trained_model)