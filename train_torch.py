import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import argparse
<<<<<<< HEAD

root = os.getcwd()
images_dir = root + '/images'
num_classes = len(os.listdir(images_dir)))
=======
from tqdm import tqdm

root = os.getcwd()
images_dir = root + '/images'
num_classes = len(os.listdir(images_dir))
model_path = root + '/model_weights/weights0'
>>>>>>> review

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
<<<<<<< HEAD
    device = torch.cuda.current_device()
input_size = (224,224,3)

transform = transforms.Compose([
                                transforms.Resize(150), # Resize the short side of the image to 150 keeping aspect ratio
                                transforms.CenterCrop(150), # Crop a square in the center of the image
                                transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                                ])

=======
    device = torch.device("cpu")
input_size = (224,224,3)

transform = transforms.Compose([
                                transforms.Resize(224), # Resize the short side of the image to 150 keeping aspect ratio
                                transforms.CenterCrop(224), # Crop a square in the center of the image
                                transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                                ])

class custom_resnet(nn.Module):
    def __init__(self, num_classes=1):
          super(custom_resnet, self).__init__()
          self.base_model = resnet50(pretrained=True)
          for param in self.base_model.parameters():
              param.requires_grad = False

          in_features = self.base_model.fc.in_features
          self.base_model.fc = nn.Linear(in_features, num_classes)
          self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        y = self.activation(x)
        return y


>>>>>>> review
def train_model(model, optimizer, criterion, train_loader, epochs):

    for epoch in range(epochs):
        # train
        model.train()
        train_loop = tqdm(train_loader, unit=" batches")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
            #print(output, target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-o', '--objects', default=None, help='classes to train')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-ld', '--logdir', default=(os.getcwd() + r'/tf_logs'), help='Location of saved tf.summary')
    args = parser.parse_args()
<<<<<<< HEAD
    
    model = resnet50(pretrained=True)
    for param in model.parameters():
        para.requires_grad = False
    ftrs = model.fc.in_features
    model.fc = nn.Linear(features, num_classes)
    
    train_dataset = ImageFolder(images_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()
    train_model(net, optimizer, loss, train_loader, args.num_epochs)
    
    
=======

    train_dataset = ImageFolder(images_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = custom_resnet()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.BCELoss()
    train_model(model, optimizer, loss, train_loader, args.num_epochs)

    torch.save(model.state_dict(), model_path)
>>>>>>> review
