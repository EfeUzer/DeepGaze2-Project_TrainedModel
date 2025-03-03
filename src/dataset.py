import os
import numpy as np
import pandas as pd
import cv2
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from datetime import datetime


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

        print(f"Datenverzeichnis: {data_root}")

        for person_id, folder in enumerate(persons):
            person_path = os.path.join(data_root, folder)
            log_data_path = os.path.join(person_path, "log_data.csv")
            gaze_data_path = os.path.join(person_path, f"gazeData{str(person_id + 1).zfill(2)}.csv")

            print(f"Überprüfe {person_path}")
            if not os.path.exists(log_data_path):
                print(f"WARNUNG: {log_data_path} nicht gefunden, übersprungen.")
                continue
            if not os.path.exists(gaze_data_path):
                print(f"WARNUNG: {gaze_data_path} nicht gefunden, übersprungen.")
                continue

            # Lade Log-Daten und konvertiere Zeitstempel in Unix-Zeit
            log_data = pd.read_csv(log_data_path)
            log_data['time'] = log_data['time'].apply(
                lambda x: datetime.fromisoformat(x[:-6]).timestamp() * 1000)  # In ms umrechnen
            log_data['time'] = log_data['time'].astype(int)
            log_data['screenshot'] = log_data['screenshot'].apply(
                lambda x: os.path.basename(x))  # Nur Dateiname extrahieren

            # Lade Gaze-Daten und stelle sicher, dass Zeitstempel Integer sind
            gaze_data = pd.read_csv(gaze_data_path)
            gaze_data['timestamp'] = gaze_data['timestamp'].astype(int)

            # Verknüpfe Screenshots mit Blickdaten
            for _, log_row in log_data.iterrows():
                screenshot_name = log_row['screenshot']  # Nur Dateiname
                image_path = os.path.join(person_path, screenshot_name)

                if os.path.exists(image_path):
                    # Finde den passendsten Blickpunkt für diesen Screenshot
                    matching_gaze = gaze_data.iloc[(gaze_data['timestamp'] - log_row['time']).abs().argsort()[:1]]
                    if not matching_gaze.empty:
                        x, y = matching_gaze.iloc[0][['x', 'y']]

                        import random

                        import random

                        # Finde die nächsten 50 Blickpunkte um den Timestamp herum
                        matching_gaze = gaze_data.iloc[(gaze_data['timestamp'] - log_row['time']).abs().argsort()[:50]]

                        # Falls es Daten gibt, wähle einen zufälligen Eintrag aus den 50 nächsten
                        if not matching_gaze.empty:
                            random_index = random.randint(0, len(matching_gaze) - 1)
                            x, y = matching_gaze.iloc[random_index][['x', 'y']]

                        # Wähle zufällig eine Zeile aus den 10 nächstgelegenen Zeitstempeln
                        if not matching_gaze.empty:
                            random_index = random.randint(0, len(matching_gaze) - 1)
                            x_original = float(matching_gaze.iloc[random_index]['x'])
                            y_original = float(matching_gaze.iloc[random_index]['y'])

                            # Skalierung auf 320x160
                            x = int(x_original * (320 / 1920))
                            y = int(y_original * (160 / 1080))

                            # Debugging: Werte ausgeben
                            print(
                                f"Zufällige Auswahl -> Original X: {x_original}, Skaliertes X: {x}, Original Y: {y_original}, Skaliertes Y: {y}")

                            self.data.append((image_path, int(x), int(y)))

                        self.data.append((image_path, int(x), int(y)))
                else:
                    print(f"WARNUNG: Bild {image_path} nicht gefunden.")

        print(f"Gesamtanzahl geladener Daten: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, x, y = self.data[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0  # Normalisierung
        height, width, _ = image.shape

        # Begrenze x und y auf die Bildgröße
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))

        if self.transform:
            image = self.transform(image)

        heatmap = np.zeros((height, width))
        heatmap[y, x] = 1  # Blickpunkt setzen
        heatmap = gaussian_filter(heatmap, sigma=15)  # Weiche Heatmap

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # (C, H, W)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

        return image, heatmap, img_path


# Beispiel für die Verwendung
data_root = os.path.expanduser("~/Downloads/DATA")
person_folders = ["19_14", "19_59", "20_17", "20_40"]

dataset = GazeDataset(data_root, person_folders)
print("Dataset-Instanz erstellt!")
if len(dataset) == 0:
    print("FEHLER: Keine Daten geladen! Bitte überprüfe die Dateipfade.")
else:
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Wähle zufällige Probe
    random_idx = random.randint(0, len(dataset) - 1)
    image_sample, heatmap_sample, img_path = dataset[random_idx]

    # Konvertiere das Bild und die Heatmap zurück für die Anzeige
    image_sample_np = image_sample.permute(1, 2, 0).numpy()
    heatmap_sample_np = heatmap_sample.squeeze(0).numpy()

    # Heatmap skalieren für bessere Sichtbarkeit
    heatmap_sample_np = (heatmap_sample_np - heatmap_sample_np.min()) / (
                heatmap_sample_np.max() - heatmap_sample_np.min())

    # Zeige das Originalbild und die Heatmap nebeneinander an
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_sample_np)
    axes[0].set_title("Original Screenshot")
    axes[0].axis("off")

    axes[1].imshow(image_sample_np, alpha=0.5)
    axes[1].imshow(heatmap_sample_np, cmap='jet', alpha=0.5)
    axes[1].set_title("Vorhergesagte Heatmap")
    axes[1].axis("off")

    plt.show()
