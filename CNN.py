import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


def train_model(args,
                model,
                dataloader,
                optimizer,
                entropy_loss):

    loss_list = []
    # Train CNN
    for epoch in range(0, args.epochs + 1):  # loop over the dataset multiple times

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

        #scheduler.step()

    print('Finished Training')
    cnn_path = './forth_cnn.pth'
    torch.save(model.state_dict(), cnn_path)

    return loss_list


def test_model(model,
               dataloader):

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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, class_labels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
