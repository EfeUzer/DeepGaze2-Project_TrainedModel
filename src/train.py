import os
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter


class GazeDataset(Dataset):
    def __init__(self, data_root, persons, transform=None):
        """
        Initialisiert das Dataset.
        :param data_root: Verzeichnis mit allen Personendaten
        :param persons: Liste der Ordnernamen für jede Person
        :param transform: Bildtransformationen (falls benötigt)
        """
        self.data = []
        self.transform = transform

        for person_id, folder in enumerate(persons):
            person_path = os.path.join(data_root, folder)
            log_data_path = os.path.join(person_path, "log_data.csv")
            gaze_data_path = os.path.join(person_path, f"gazeData0{person_id + 1}.csv")

            if not os.path.exists(log_data_path) or not os.path.exists(gaze_data_path):
                print(f"WARNUNG: Datei fehlt in {person_path}, wird übersprungen.")
                continue

            # Lade Log-Daten
            log_data = pd.read_csv(log_data_path)
            gaze_data = pd.read_csv(gaze_data_path)

            # Verknüpfe Screenshots mit Blickdaten
            for _, log_row in log_data.iterrows():
                screenshot_name = log_row['screenshot']
                image_path = os.path.join(person_path, screenshot_name)

                if os.path.exists(image_path):
                    # Finde den passenden Blickpunkt für diesen Screenshot
                    matching_gaze = gaze_data[gaze_data['timestamp'] == log_row['time']]
                    if not matching_gaze.empty:
                        x, y = matching_gaze.iloc[0][['x', 'y']]
                        self.data.append((image_path, x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, x, y = self.data[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalisierung

        if self.transform:
            image = self.transform(image)

        heatmap = np.zeros((image.shape[0], image.shape[1]))
        heatmap[int(y), int(x)] = 1  # Blickpunkt setzen
        heatmap = gaussian_filter(heatmap, sigma=15)  # Weiche Heatmap

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return image, heatmap


# Beispiel für die Verwendung
data_root = os.path.expanduser("~/Downloads/DATA")
person_folders = ["19_14", "19_59", "20_17", "20_40"]

dataset = GazeDataset(data_root, person_folders)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Testen
data_iter = iter(dataloader)
image_sample, heatmap_sample = next(data_iter)
print("Image Shape:", image_sample.shape)
print("Heatmap Shape:", heatmap_sample.shape)
