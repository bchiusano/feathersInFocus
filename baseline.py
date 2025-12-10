import pandas as pd
import numpy as np
from PIL import Image
import torch
from transformers import ViTForImageClassification
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F
from transformers import ViTConfig
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



# # # # INFORMATION
# # # Dataset contains 200 bird species 
# # # submit a csv file with the image name and the prediced class label
# # # Order of the rows does not matter
# # # labels go from 1-200


# # # resized/rescaled to the same resolution (224x224)
# # # normalized across the RGB channels with mean (0.5, 0.5, 0.5)
# # # and standard deviation (0.5, 0.5, 0.5)


# load data
train_set = pd.read_csv('train_images.csv')
test_set = pd.read_csv('test_images_path.csv')

# load classes and attributes
class_names = np.load("class_names.npy", allow_pickle=True).item()
n_classes = len(class_names)

attributes = np.load("attributes.npy", allow_pickle=True)


config = ViTConfig(
    image_size=224,          
    patch_size=16,
    hidden_size=64,         
    intermediate_size=128, 
    num_hidden_layers=2,    
    num_attention_heads=2,  
    num_channels=3,
    num_labels=n_classes,
)
model = ViTForImageClassification(config)


# # # ok now working on the class that will actually do the work from pytorch basics MINST example

transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5],
                     std=[0.5, 0.5, 0.5])])

class ImageClassification(Dataset):
    def __init__(self, df):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = "train_images/" + row["image_path"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label = int(row["label"]) - 1

        return image, label


# splitting 
train_df = pd.read_csv('train_images.csv')
train_split, test_split = train_test_split(train_df, test_size=0.2, random_state=42)
train_dataset = ImageClassification(train_split)
test_dataset = ImageClassification(test_split)

# dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# making sure everything is on the same device

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# optimizer 

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

## first train and test without features

def train_one_epoch(model, loader):
    model.train()
    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(pixel_values=images)
        loss = F.cross_entropy(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values=images)
            preds = outputs.logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    print("Training baseline ViT (from scratch) for 1 epoch...")
    train_one_epoch(model, train_loader)

    print("Evaluating on test split...")
    baseline_acc = evaluate(model, test_loader)
    print("Baseline accuracy:", baseline_acc)


# creatinc csv for kaggle 

sample = pd.read_csv("test_images_sample.csv")
predictions = []
model.eval()
with torch.no_grad():
    for image_name in sample["id"]:
        image_name = str(image_name).strip() + ".jpg"
        image = Image.open(f"test_images/test_images/{image_name}").convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)

        prediction = model(pixel_values=image).logits.argmax(dim=1).item() + 1 # i think this is needed because before we did -1 to have 0-199 but now need 1-200 again?
        predictions.append(prediction)


sample["label"] = predictions
sample.to_csv("baseline.csv", index=False)
print("baseline.csv created")










# # # then train and test with features

# # class ImageClassificationWithAttributes(Dataset):
# #     def __init__(self, df):
# #         self.df = df
# #         self.transform = transform
        
# #     def __len__(self):
# #         return len(self.df)
    
# #     def __getitem__(self, idx):
# #         row = self.df.iloc[idx]
# #         image_path = "train_images/" + row["image_path"]
# #         image = Image.open(image_path).convert("RGB")
# #         image = self.transform(image)
# #         label = int(row["label"]) - 1

# #         attr = torch.tensor(attributes[label], dtype=torch.float32)

# #         return image, attr, label


# # # splitting 
# # train_df = pd.read_csv('train_images.csv')
# # train_split, test_split = train_test_split(train_df, test_size=0.2, random_state=42)
# # train_dataset = ImageClassificationWithAttributes(train_split)
# # test_dataset = ImageClassificationWithAttributes(test_split)

# # # dataloader

# # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
# # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# # # wrapper class with attributes

# # class ViTWithAttributes(torch.nn.Module):
# #     def __init__(self, vit_model, attr_dim, n_classes):
# #         super().__init__()
# #         self.vit = vit_model

# #         # vit_model outputs logits of shape (B, n_classes)
# #         vit_output_dim = n_classes

# #         # new classifier head
# #         self.classifier = torch.nn.Linear(vit_output_dim + attr_dim, n_classes)

# #     def forward(self, images, attrs):
# #         # image-only logits
# #         vit_out = self.vit(pixel_values=images).logits  # (B, n_classes)

# #         # concatenate image logits + attribute vector
# #         combined = torch.cat([vit_out, attrs], dim=1)

# #         logits = self.classifier(combined)
# #         return logits

# # # device

# # DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # model = ViTWithAttributes(model, attr_dim, n_classes).to(DEVICE)

# # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # # training loop
# # def train_one_epoch(model, loader):
# #     model.train()
# #     total_loss = 0

# #     for images, attrs, labels in loader:
# #         images = images.to(DEVICE)
# #         attrs = attrs.to(DEVICE)
# #         labels = labels.to(DEVICE)

# #         logits = model(images, attrs)
# #         loss = F.cross_entropy(logits, labels)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# #     return total_loss / len(loader)

# # # evaluation loop

# # def evaluate(model, loader):
# #     model.eval()
# #     correct = 0
# #     total = 0

# #     with torch.no_grad():
# #         for images, attrs, labels in loader:
# #             images = images.to(DEVICE)
# #             attrs = attrs.to(DEVICE)
# #             labels = labels.to(DEVICE)

# #             logits = model(images, attrs)
# #             preds = logits.argmax(dim=1)

# #             correct += (preds == labels).sum().item()
# #             total += labels.size(0)

# #     return correct / total


# # if __name__ == "__main__":
# #     print("Training ViT + Attributes for 1 epoch...")
# #     train_loss = train_one_epoch(model, train_loader)
# #     print(f"Train Loss: {train_loss:.4f}")

# #     print("Evaluating on test split...")
# #     acc = evaluate(model, test_loader)
# #     print("Accuracy:", acc)