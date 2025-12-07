import pandas as pd
import torch as tc
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import os


TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224)),
        v2.ToDtype(tc.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])
    ]
)


class DataLoad(Dataset):

    def __init__(self,
                 data,
                 data_path: Path,
                 transform: v2.Transform = TRANSFORM):

        self.data = data
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item_idx):
        # get image path
        image_path = os.path.join(self.data_path, self.data['image_path'][item_idx].lstrip('/'))
        # read the image and convert it to RGB and apply transformation
        image = self.transform(Image.open(image_path).convert("RGB"))

        label_image = int(self.data['label'][item_idx])

        return image, label_image


