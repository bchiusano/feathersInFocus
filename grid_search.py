import pandas as pd
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier # skorch makes pytorch work with sklearn,
                                    # provides wrapper around PyTorch that has an sklearn interface
import torch
import torch.nn as nn
from pathlib import Path

from CNN import CNN
from DataLoad import DataLoad


train_df = pd.read_csv('train_images.csv')
X = train_df['image_path'].values
y = train_df['label'].values

# skorch expect X, but dataset loads images from path
# so wrapping DataLoad makes it skorch-friendly
class BirdDataset(torch.utils.data.Dataset):
    def __init__(self,X,y):  
        df = pd.DataFrame({
            'image_path': X,
            'label': y
        })          
        self.dataset = DataLoad(data=df, data_path = Path('./train_images'))

    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        return self.dataset[index]
    
# skorch
net = NeuralNetClassifier(
    module=CNN, 
    module__class_labels = 201,
    criterion=nn.CrossEntropyLoss,
    device='cpu',
    dataset = BirdDataset
)

# search space
#param_grid = {   THIS IS THE FIRST PARAM_GRID TO FIND LR, USED ADAM DID NOT USE BATCH SIZE
 #   'lr': [0.0001, 0.001],
  #  'max_epochs': [3], # should be more in real training
   # #'batch_size': [32, 64],
    #'optimizer': [torch.optim.AdamW], # try sgd optimizer 
    #'optimizer__weight_decay': [0.05]
#}
best_lr = 0.0001 # from initial param grid
param_grid = {
    'lr' : [best_lr],
    'max_epochs': [5], # 3
    #'batch_size': [32,64],
    'optimizer': [torch.optim.AdamW],
    'optimizer__weight_decay':[0.01, 0.05]
}
grid = GridSearchCV(
    estimator=net,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    refit=True,
    verbose=2
)

grid.fit(X,y)

print(grid.best_params_)

print(grid.best_score_)