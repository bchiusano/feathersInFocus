import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import argparse

from DataLoad import DataLoad
from CNN import CNN
from CNN import train_model


def load_data():
    # TODO: im guessing this is more for visualisation purposes
    class_names = np.load("class_names.npy", allow_pickle=True).item()

    # load the data with pandas
    train_set = pd.read_csv('train_images.csv')
    test_set = pd.read_csv('test_images_path.csv')

    train_dataset = DataLoad(data=train_set, data_path=Path("./train_images"))
    test_dataset = DataLoad(data=test_set, data_path=Path("./test_images"))

    return train_dataset, test_dataset, class_names


# Specify the hyperparameters
def load_arg_parser():
    parser = argparse.ArgumentParser(description="Feathers in Focus")
    parser.add_argument("--batch-size", type=int, default=64)
    # TODO: not sure what the batch size should be for the test
    parser.add_argument("--test-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-model", action="store_true")
    return parser


if __name__ == "__main__":
    DEBUG = False
    # load arg parser
    parser = load_arg_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # load and transform the data
    train_data, test_data, classes = load_data()

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

    # CNN and Calculate Loss Function
    cnn = CNN(class_labels=201).to(device)

    entropy_loss = nn.CrossEntropyLoss()  # TODO: which loss to use
    optimizer = optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.9)  # TODO: which optimizer to use
    # halves the learning rate every 5 epochs
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    train_model(args,
                cnn,
                train_dataloader,
                optimizer,
                entropy_loss,
                scheduler)
