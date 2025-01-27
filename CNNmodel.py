#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR

"""

I added transformations to the training data set to improve generalisation in the model.
Flips and rotations were put in because cats themselves can be lots of different orientations.
Colour adjustments were made because some cat breeds varied highly in colour so I wanted to emphasise
texture and pattern recognition over colour. Brightness, contrast, and saturation were also adjusted
to a smaller scale to account for variation in image quality and lighting differences.

Weight decay was added in the optimiser and scheduler to improve generalisation and to reduce large weights.

Cross entropy was used as a loss function because of its suitablity for multiclassification problems.

Weight initialisation was used to improve training speed mainly. Xavier uniform in particular due to its suitability
with ReLu activations.

Convolutional layer depth choice was managing the trade off between computational cost and performance.

Max pool after first layer to reduce feature map of initial input for processing purposes.

Dropouts in the fully connected layers to improve generalisation, 
removed from convolutional layers to not inhibit pattern recognition.

"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):

    if mode == 'train':
        return transforms.Compose([
            transforms.RandomVerticalFlip(p=0.25),
            transforms.RandomHorizontalFlip(p=0.25),
            transforms.RandomRotation(60),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.5),
#            transforms.RandomResizedCrop(80, scale=(0.8, 1.0)),
            transforms.ToTensor(),
        ])
    elif mode == 'test':
        return transforms.Compose([
            transforms.ToTensor(),
        ])

#    if mode == 'train':
#        return transforms.ToTensor()
#    elif mode == 'test':
#        return transforms.ToTensor()

# Transformation only applied to training data to not overrepresent training performance by making training data like testing

############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=1, padding=1) #out_channel 2x output and each conv layer double
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.hid1 = nn.Linear(256 * 19 * 19, 128) #19*19 due to max pool reduction and kernel 5
        self.drop1 = nn.Dropout(0.5)

        self.hid2 = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.5)

        self.output = nn.Linear(64, 8) #output layer of 8 classifcations

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        x = x.view(-1, 256 * 19 * 19)

        x = F.relu(self.hid1(x))
        x = self.drop1(x)

        x = F.relu(self.hid2(x))
        x = self.drop2(x)

        x = self.output(x)

        x = F.log_softmax(x, dim=1)

        return x


net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.0001) #, 
#optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.5)

#loss_func = F.nll_loss
loss_func = F.cross_entropy


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    return

#scheduler = None
scheduler = StepLR(optimizer, step_size=5, gamma=0.9) #gamma 0.9 to not increase weight decay in optimiser too much

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 200
epochs = 1000
