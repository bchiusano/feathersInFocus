import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def train_model(args,
                model,
                dataloader,
                optimizer,
                scheduler,
                entropy_loss):

    model.train()
    loss_list = []
    # Train CNN
    for epoch in range(0, args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, target = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            # calculating the loss, this also performs softmax
            loss = entropy_loss(outputs, target)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        # Print once per epoch
        avg_loss = running_loss / len(dataloader)
        print(f'[Epoch {epoch + 1}] average loss: {avg_loss:.3f}')
        loss_list.append(avg_loss)

        scheduler.step()

    print('Finished Training')
    cnn_path = './birds_cnn.pt'
    torch.save(model.state_dict(), cnn_path)

    return loss_list


def test_model(model,
               dataloader):
    # does not perform dropout
    model.eval()

    predicted_labels = []
    sample = pd.read_csv('test_images_sample.csv')

    with torch.no_grad():
        for data in dataloader:
            inputs, targets = data
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for prediction in predicted:
                # reindexing to 1
                predicted_labels.append(int(prediction) + 1)

    sample['label'] = predicted_labels
    sample.to_csv('test_predictionsNew.csv', index=False)


class CNN(nn.Module):
    def __init__(self, class_labels):
        super().__init__()

        self.extract_features = nn.Sequential(
            # first convolution
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # second convolution
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # third convolution
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # forth convolution
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classify = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, class_labels),
        )

    def forward(self, x):
        x = self.extract_features(x)
        x = torch.flatten(x, 1)
        x = self.classify(x)
        return x
