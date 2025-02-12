import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from timm.models.swin_transformer import SwinTransformer
from torchvision import datasets, transforms
import numpy as np
import os
import cv2
import random
import shutil
from tqdm import tqdm
from torch.autograd import Variable
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Define hyperparameters
batch_size = 256
learning_rate = 0.001
epochs = 50

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load data
data = datasets.ImageFolder('data', transform=transform)
train_data, test_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), len(data) - int(len(data) * 0.8)])
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
swin_tiny_cfg = dict(patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
swin_tiny = SwinTransformer(**swin_tiny_cfg, num_classes=6)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.backbone = SwinTransformer(**swin_tiny_cfg)
        self.backbone.load_state_dict(
            torch.load('swin_tiny_patch4_window7_224.pth',
                       map_location=torch.device(device))['model'], strict=True)
         # 此时维度为([batchsize，49,768])
        self.fc1 = nn.Linear(49*768, 256)
        self.fc2 = nn.Linear(256, 6)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Net()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
train_losses = []
train_counter = []
test_losses = []
# 训练过程
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).to(device), Variable(target).to(device)
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item()))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))


# 验证过程
def val(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))


# 训练

for epoch in range(1, epochs + 1):
    adjust_learning_rate(optimizer, epoch)
    train(model, device, train_loader, optimizer, epoch)
    val(model, device, test_loader)
# 保存模型
torch.save(model.state_dict(), 'swin.pth')

