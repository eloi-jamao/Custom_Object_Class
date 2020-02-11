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
from tqdm import tqdm

root = os.getcwd()
images_dir = root + '/images'
test_dir = root + '/test'
num_classes = len(os.listdir(images_dir))
model_path = root + f'/model_weights/weights' + str(len(os.listdir(root+'/model_weights')))
input_size = (224,224,3)

if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")

transform = transforms.Compose([
                                transforms.Resize(input_size[0]), # Resize the short side of the image to 150 keeping aspect ratio
                                transforms.CenterCrop(input_size[0]), # Crop a square in the center of the image
                                transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                                ])

class custom_resnet(nn.Module):
    def __init__(self, num_classes=num_classes):
          super(custom_resnet, self).__init__()
          self.base_model = resnet50(pretrained=True)
          for param in self.base_model.parameters():
              param.requires_grad = False

          in_features = self.base_model.fc.in_features
          self.base_model.fc = nn.Linear(in_features, num_classes)
          print(self.base_model)

    def forward(self, x):
        y = self.base_model(x)

        return y

def train_model(model, optimizer, criterion, train_loader, epochs):

    for epoch in range(epochs):
        # train
        model.train()
        train_loop = tqdm(train_loader, unit=" batches")  # For printing the progress bar
        for data, target in train_loop:
            train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch + 1, epochs))
            data, target = data.float().to(device), target.to(device)
            #target = target.unsqueeze(-1)
            optimizer.zero_grad()
            output = model(data)
            #print(output.shape, target.shape)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def test_model(model,test_loader, epochs):

    for epoch in range(epochs):
        # test
        acc = 0
        model.eval()
        test_loop = tqdm(test_loader, unit="batches")  # For printing the progress bar
        with torch.no_grad():
            for data, target in test_loop:
                test_loop.set_description('[TEST] Epoch {}/{}'.format(epoch + 1, epochs))
                data, target = data.float().to(device), target.to(device)
                #target = target.unsqueeze(-1)
                output = model(data)
                #print(output.shape, target.shape)
                acc += correct_predictions(output, target)
        acc = 100. * acc / len(test_loader.dataset)
        print(f'Test accuracy of {acc}')

def correct_predictions(predicted_batch, label_batch):
  pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
  acum = pred.eq(label_batch.view_as(pred)).sum().item()
  return acum

def describe_dataset(dataset):
    it=iter(dataset)
    for image,label in it:
        print(len(dataset))
        print(label)
        plt.imshow(image.T)
        plt.show()
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pipeline execution')
    parser.add_argument('-o', '--objects', default=None, help='classes to train')
    parser.add_argument('-e', '--num_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=5, help='Batch size')
    parser.add_argument('-t', '--test', default=False, help='Test the model during training')
    parser.add_argument('-ld', '--logdir', default=(os.getcwd() + r'/tf_logs'), help='Location of saved tf.summary')
    args = parser.parse_args()

    train_dataset = ImageFolder(images_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.test:
        test_dataset = ImageFolder(test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=1)

    #describe_dataset(train_dataset)

    loss = nn.CrossEntropyLoss()
    model = custom_resnet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss = nn.CrossEntropyLoss()

    train_model(model, optimizer, loss, train_loader, args.num_epochs)
    if args.test:
        test_model(model, test_loader, args.num_epochs)

    torch.save(model.state_dict(), model_path)
