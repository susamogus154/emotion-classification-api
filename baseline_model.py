import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import io

class EmotionClassifier(nn.Module):
  def __init__(self, num_classes=7):
    super(EmotionClassifier, self).__init__()
    # self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
    # self.features = nn.Sequential(*list(self.base_model.children())[:-1]) # removing the final layer
    # enet_out_size = 1280 # output size of the self.features
    # self.classifier = nn.Linear(enet_out_size, num_classes) # final layer for this amount of classes specifically


    self.features = nn.Sequential(
        # block 1 - 128x79 -> 64x39
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # block 2 - 64x39 -> 32x19
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # block 3 - 32x19 -> 16x9
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # block 4 - 16x9 -> 8x4
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 8 * 4, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes),
    )


  def forward(self, x):
    x = self.features(x) # feed through first section of model (efficientnet)
    output = self.classifier(x) # feed through final classification layer
    # print("p repuires grad: ", sum(p.requires_grad for p in model.parameters()))
    return output

  def predict(self, image_tensor):
    self.eval()
    with torch.no_grad():
      outputs = self.forward(image_tensor) # forward function
      probabilities = torch.nn.functional.softmax(outputs, dim=1) # final softmax activation function
       
    return probabilities.numpy().flatten()