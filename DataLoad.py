import pandas as pd
import torch as tc
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
import os
import matplotlib.pyplot as plt


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


DEBUG = False
DATA_TRAIN_DIR = Path("./train_images")
DATA_TEST_DIR = Path("./test_images")

# im guessing this is more for visualisation purposes
# class_names = np.load("class_names.npy", allow_pickle=True).item()

# load the data with pandas
train_set = pd.read_csv('train_images.csv')
test_set = pd.read_csv('test_images_path.csv')

train_dataset = DataLoad(data=train_set, data_path=DATA_TRAIN_DIR)
test_dataset = DataLoad(data=test_set, data_path=DATA_TEST_DIR)

if DEBUG:
    transformed_image, label = train_dataset.__getitem__(0)
    print(label)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")