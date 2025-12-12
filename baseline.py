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
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import seaborn as sns
from torchinfo import summary
from sklearn.metrics import confusion_matrix



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

# load classes and attributes
class_names = np.load("class_names.npy", allow_pickle=True).item()
print(class_names)
n_classes = len(class_names)


config = ViTConfig(
    image_size=224,          
    patch_size=16,
    hidden_size=256,         
    intermediate_size=512, 
    num_hidden_layers=6,    
    num_attention_heads=8,  
    num_channels=3,
    num_labels=n_classes,
)
model = ViTForImageClassification(config)


transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

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
train_split, test_split = train_test_split(train_set, test_size=0.2, random_state=42)
train_dataset = ImageClassification(train_split)
test_dataset = ImageClassification(test_split)

# dataloader

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True) 
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# making sure everything is on the same device

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# optimizer 

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, train_loader, test_loader, epochs=3):
    epoch_times = []
    train_losses = []
    test_accuracy = []
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values=images)
            loss = F.cross_entropy(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)
        train_losses.append(total_loss / len(train_loader))
        train_acc = evaluate(model, train_loader)
        test_acc = evaluate(model, test_loader)
        test_accuracy.append(test_acc)

        print(f"epoch {epoch+1}/{epochs}, loss: {total_loss/len(train_loader):.4f} "
              f"training accuracy: {train_acc:.4f}, test accuracy: {test_acc:.4f}", f"Time: {epoch_time:.2f}s")
        
    return train_losses, test_accuracy, epoch_times


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(pixel_values=images)
            preds = outputs.logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total


if __name__ == "__main__":
    print("Training baseline ViT")
    train(model, train_loader, test_loader)

    print("Evaluating on test..")
    baseline_acc = evaluate(model, test_loader)
    print("baseline accuracy:", baseline_acc)

# some metrics for computational efficiency comparison to main model

info = summary(model, input_size=(1, 3, 224, 224), verbose=0)
total_parameters = sum(p.numel() for p in model.parameters())
print("Total FLOPs:", info.total_mult_adds * 2)
print("Total parameters: ", total_parameters)

# visualizing what is going on inside the model 

model.eval()
for i in range(2):
    image, label = test_dataset[i]
    image_tensor = image.unsqueeze(0).to(DEVICE)

    outputs = model(pixel_values=image_tensor)
    logits = outputs.logits.squeeze()
    probs = torch.softmax(logits, dim=0)

    pred_class = probs.argmax().item()

    print(f"Image {i+1}")
    print(f"True label: {label+1}")
    print(f"Logits (first 10 classes): {logits[:10].detach().cpu().numpy()}")
    print(f"Probabilities (first 10 classes): {probs[:10].detach().cpu().numpy()}")
    print(f"Predicted class: {pred_class+1} (prob={probs[pred_class].item():.4f})")


# creatinc csv for kaggle 

# sample = pd.read_csv("test_images_sample.csv")
# predictions = []
# model.eval()
# with torch.no_grad():
#     for image_name in sample["id"]:
#         image_name = str(image_name).strip() + ".jpg"
#         image = Image.open(f"test_images/test_images/{image_name}").convert("RGB")
#         image = transform(image).unsqueeze(0).to(DEVICE)

#         prediction = model(pixel_values=image).logits.argmax(dim=1).item() + 1 # i think this is needed because before we did -1 to have 0-199 but now need 1-200 again
#         predictions.append(prediction)


# sample["label"] = predictions
# sample.to_csv("baseline.csv", index=False)
# print("baseline.csv created")


