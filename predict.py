import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
import argparse
from PIL import Image

root = os.getcwd()
images_dir = root + '/images'
num_classes = len(os.listdir(images_dir))
weights_path = root + '/model_weights' + '/weights0'

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

def image_loader(image_name):
    """load image, returns tensor"""
    image = Image.open(image_name)
    image = transform(image).float()
    #image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('-i', '--image', help='image to predict')
    args = parser.parse_args()

    model = custom_resnet()
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    img_path = images_dir + '/grinder/' + args.image
    data = image_loader(img_path)
    preds = model(data)
    print(preds)