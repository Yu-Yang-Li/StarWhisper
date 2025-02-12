import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import time
import os
import math
import matplotlib.pyplot as plt

# use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device='cuda'

# Read data from csv file
data = pd.read_csv('merged.csv')

# Get labels and features
labels = data.iloc[:, 0].values
features = data.iloc[:, 1:].values

#%%
# normalize the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

# split dataset into train and val
from sklearn.model_selection import train_test_split
train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size=0.2, random_state=0)



#%%
# create dataloader
from torch.utils.data import TensorDataset, DataLoader
train_data = TensorDataset(torch.from_numpy(train_labels).long(), torch.from_numpy(train_features).float())
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_data = TensorDataset(torch.from_numpy(val_labels).long(), torch.from_numpy(val_features).float())
val_loader = DataLoader(val_data, batch_size=128, shuffle=False)


#%%
# define dataset
class LCDataset(Dataset):
    def __init__(self, label, flux):
        self.label = label
        self.flux = flux

    def __getitem__(self, index):
        label = self.label[index]
        flux = self.flux[index]
        return label, flux

    def __len__(self):
        return len(self.label)
# use optuna to find the best hyperparameters（including learning rate, number of layers, number of filters, dropout rate, etc.）
import optuna
from optuna.samplers import TPESampler

# define the model
class LCModel(nn.Module):
    def __init__(self, trial):
        super(LCModel, self).__init__()
        self.trial = trial
        self.num_layers = trial.suggest_int('num_layers', 1, 3)
        self.num_filters = trial.suggest_int('num_filters', 32, 128)
        self.dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)
        self.conv1 = nn.Conv1d(1, self.num_filters, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(self.num_filters, self.num_filters*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(self.num_filters*2, self.num_filters*4, kernel_size=3, stride=1, padding=1)
        self.lstm = nn.LSTM(self.num_filters*4, 128, self.num_layers, batch_first=True, bidirectional=True, dropout=self.dropout_rate)
        self.fc1 = nn.Linear(256, 64)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.fc2 = nn.Linear(64, 6)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
#%%
# define objective function
def objective(trial):
    model = LCModel(trial).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=trial.suggest_loguniform('lr', 1e-5, 1e-1))
    num_epochs = 20
    for epoch in range(num_epochs):
        model.train()
        for i, (label, flux) in enumerate(train_loader):
            label = label.to(device)
            flux = flux.to(device)
            optimizer.zero_grad()
            outputs = model(flux)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for label, flux in val_loader:
                label = label.to(device)
                flux = flux.to(device)
                outputs = model(flux)
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            accuracy = correct / total
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                    .format(epoch + 1, num_epochs, loss.item(), (correct / total) * 100))
    return accuracy

# define study
study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=0))
study.optimize(objective, n_trials=20)


# print the best hyperparameters
print('best hyperparameters: {}'.format(study.best_params))

#%%
## optuna visualization
# optuna.visualization.plot_optimization_history(study)
# optuna.visualization.plot_slice(study)
# optuna.visualization.plot_contour(study)
# optuna.visualization.plot_parallel_coordinate(study)
# optuna.visualization.plot_intermediate_values(study)
# optuna.visualization.plot_edf(study)
# save the best model

best_model = LCModel(study.best_trial).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(best_model.parameters(), lr=study.best_params['lr'])
num_epochs = 100
val_accuracy_list = []
train_loss_list = []
val_loss_list = []
for epoch in range(num_epochs):
    best_model.train()
    for i, (label, flux) in enumerate(train_loader):
        label = label.to(device)
        flux = flux.to(device)
        optimizer.zero_grad()
        outputs = best_model(flux)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        train_loss_list.append(loss.item())
    best_model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        correct = 0
        total = 0
        for label, flux in val_loader:
            label = label.to(device)
            flux = flux.to(device)
            outputs = best_model(flux)
            _, predicted = torch.max(outputs.data, 1)
            y_true.append(label.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
            total += label.size(0)
            correct += (predicted == label).sum().item()
        accuracy = correct / total
        print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                .format(epoch + 1, num_epochs, loss.item(), (correct / total) * 100))
        val_loss_list.append(loss.item())
        val_accuracy_list.append(accuracy)
torch.save(best_model.state_dict(), 'best_model.pth')
# # %%
# # visualize epoch-loss,epoch-accuracy and train-val loss
# from sklearn.metrics import confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # Create figure with two subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
#
# # Plot train_loss_list and val_loss_list in ax1
# ax1.plot(train_loss_list, label='train_loss')
# ax1.plot(val_loss_list, label='val_loss')
# ax1.set_xlabel('epoch')
# ax1.set_ylabel('value')
# ax1.legend()
#
# # Plot val_accuracy_list and val_loss_list in ax2
# ax2.plot(val_accuracy_list, label='accuracy')
# ax2.plot(val_loss_list, label='val_loss')
# ax2.set_xlabel('epoch')
# ax2.set_ylabel('value')
# ax2.legend()
#
# plt.show()
# # %%
# # viasualize confusion matrix
# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt="d")
# plt.title("Confusion matrix", fontsize=20)
# plt.ylabel('True label', fontsize=14)
# plt.xlabel('Predicted label', fontsize=14)
# plt.show()
# # # # %%
# # # # test the best model
# # # best_model.load_state_dict(torch.load('best_model.pth'))
# # # best_model.eval()
# # # with torch.no_grad():
# # #     correct = 0
# # #     total = 0
# # #     for label, flux in test_loader:
# # #         label = label.to(device)
# # #         flux = flux.to(device)
# # #         outputs = best_model(flux)
# # #         _, predicted = torch.max(outputs.data, 1)
# # #         total += label.size(0)
# # #         correct += (predicted == label).sum().item()
# # #     print('Test Accuracy of the model on the test set: {} %'.format((correct / total) * 100))