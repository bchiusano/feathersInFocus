import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

from DataLoad import DataLoad
from CNN import CNN
from CNN import train_model
from CNN import test_model
from CNN import calc_accuracy


def load_data():
    # TODO: im guessing this is more for visualisation purposes
    class_names = np.load("class_names.npy", allow_pickle=True).item()

    # load the data with pandas
    train_set = pd.read_csv('train_images.csv')
    test_set = pd.read_csv('test_images_path.csv')

    train_x, validate_x, train_y, validate_y = train_test_split(train_set['image_path'],
                                                                train_set['label'],
                                                                test_size=0.2,
                                                                random_state=42)
    train = pd.DataFrame({'image_path': train_x, 'label': train_y}).reset_index(drop=True)
    validate = pd.DataFrame({'image_path': validate_x, 'label': validate_y}).reset_index(drop=True)

    #train_dataset = DataLoad(data=train_set, data_path=Path("./train_images"))
    # CHANGED
    train_dataset = DataLoad(data=train, data_path=Path("./train_images"))
    validate_dataset = DataLoad(data=validate, data_path=Path("./train_images"))
    test_dataset = DataLoad(data=test_set, data_path=Path("./test_images"))

    return train_dataset, validate_dataset, test_dataset, class_names


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
    train_data, validate_data, test_data, classes = load_data()

    # CHANGED THE SHUFFLING BECAUSE IM ALREADY DOING IT IN THE SPLIT
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    validate_dataloader = DataLoader(validate_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # CNN and Calculate Loss Function
    # TODO: not sure about these 201 classes (indexing is wrong?)
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

    # load model (once its saved) - kind of confused on whether to save it first
    cnn2 = CNN(class_labels=201)
    cnn2.load_state_dict(torch.load('./third_cnn.pth', weights_only=True))
    # test set
    # test_model(cnn2, test_dataloader)
    calc_accuracy(cnn2, validate_dataloader, classes)
