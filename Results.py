import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class Results:
    def __init__(self, name):
        class_names = np.load("class_names.npy", allow_pickle=True).item()
        print(class_names)
        self.name = name
        self.classes = class_names

    def eda_data(self, data):
        # how many of bird species in each set
        # if any species are missing

        result = data['label'].value_counts().reset_index()
        result.columns = ['label', 'count']
        # removing the dot before the class name
        reverse_classes = {v: str(k).split('.', 1)[1] for k, v in self.classes.items()}
        result['class_name'] = result['label'].map(reverse_classes)

        print(result)

    def plot_loss(self, loss):

        epochs = list(range(1, len(loss) + 1))

        plt.figure(figsize=(10, 6))
        plt.plot(epochs, loss, label=f'Training Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Average Loss', fontsize=12)
        plt.title(f'{self.name} Average Loss', fontsize=14)
        plt.tight_layout()
        plt.show()

    def calc_accuracy(self, model, dataloader):
        correct = 0
        total = 0

        # from pytorch tutorial
        correct_pred = {classname: 0 for classname in self.classes}
        total_pred = {classname: 0 for classname in self.classes}
        names = list(correct_pred.keys())

        with torch.no_grad():
            for data in dataloader:
                images, targets = data
                # calculate outputs by running images through the network
                outputs = model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # collect the correct predictions for each class
                for label, prediction in zip(targets, predicted):
                    if label == prediction:
                        correct_pred[names[label]] += 1
                    total_pred[names[label]] += 1

        print(f'Accuracy for test images: {100 * correct // total} %')

        # print accuracy for each class
        non_zero_class = []
        non_zero_acc = []
        for classname, correct_count in correct_pred.items():

            clean_name = str(classname).split('.', 1)[1]
            if total_pred[classname] != 0:
                accuracy = 100 * float(correct_count) / total_pred[classname]

                non_zero_class.append(clean_name)
                non_zero_acc.append(accuracy)
                print(f'Accuracy for class: {clean_name} is {accuracy:.1f} %')
            else:
                print(f'Accuracy for class: {clean_name} is 0%')

        # Plotting the accuracy of classes (non-zero)
        acc_df = pd.DataFrame({'class_name': non_zero_class, 'accuracy': non_zero_acc})
        acc_df = acc_df.sort_values('accuracy', ascending=False)

        # plt.figure(figsize=(14, 6))
        # sns.barplot(data=acc_df, x='class_name', y='accuracy', palette='viridis')
        # plt.xlabel('Class Name', fontsize=12)
        # plt.ylabel('Accuracy (%)', fontsize=12)
        # plt.title('Per-Class Accuracy', fontsize=14)
        # plt.xticks(rotation=90, ha='right')
        # plt.ylim(0, 100)
        # plt.tight_layout()
        # plt.show()

    def show_patterns(self):
        pass
