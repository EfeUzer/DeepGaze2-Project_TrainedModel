import os
import random
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import GazeDataset

# Lade das Dataset
data_root = os.path.expanduser("~/Downloads/DATA")
person_folders = ["19_14", "19_59", "20_17", "20_40"]
dataset = GazeDataset(data_root, person_folders)

# Wähle ein zufälliges Sample aus
random_sample = random.choice(dataset.data)
img_path, x_gt, y_gt = random_sample

# Lade das Bild
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Erstelle die Heatmap auf Basis der Ground-Truth-Punkte
heatmap_gt = np.zeros((160, 320))
heatmap_gt[int(y_gt), int(x_gt)] = 1  # Setze den Ground-Truth-Punkt
heatmap_gt = cv2.GaussianBlur(heatmap_gt, (15, 15), 0)

# Zeige das Bild mit Ground-Truth-Koordinaten
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Originalbild mit Ground-Truth-Punkt
ax[0].imshow(image)
ax[0].scatter([x_gt * (image.shape[1] / 320)], [y_gt * (image.shape[0] / 160)], color='red', label='Ground Truth', s=50)
ax[0].set_title("Originalbild mit Ground-Truth-Punkt")
ax[0].legend()

# Heatmap anzeigen
ax[1].imshow(image, alpha=0.5)
sns.heatmap(heatmap_gt, cmap="jet", alpha=0.6, ax=ax[1])
ax[1].set_title("Ground-Truth-Heatmap")

plt.show()