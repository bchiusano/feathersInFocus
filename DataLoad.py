import pandas as pd
import torch as tc
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import v2
import os

# from https://docs.pytorch.org/vision/0.22/transforms.html
TRANSFORM = v2.Compose(
    [
        v2.ToImage(),
        v2.Resize((224, 224)),

        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        v2.ToDtype(tc.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)


def calculate_img_size(image):
    pass


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
        # print("HELLO")
        # reindexing labels from 0 to 199
        label_image = int(self.data['label'][item_idx]) - 1
        # print(image_path)

        return image, label_image


