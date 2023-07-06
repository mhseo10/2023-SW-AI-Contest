import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SatelliteDataset(Dataset):
    def __init__(self, path, shape, transform=None, infer=False):
        self.data = pd.read_parquet(path)
        self.transform = transform
        self.infer = infer
        self.shape = shape
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 1]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.infer:
            if self.transform:
                image = self.transform(image=image)['image']
            return image

        mask = np.frombuffer(bytearray(self.data.iloc[idx, 2]), dtype=np.uint8).reshape(self.shape)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask