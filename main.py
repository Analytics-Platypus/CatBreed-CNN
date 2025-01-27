#!/usr/bin/env python3

import torch
import torchvision
import sklearn.metrics as metrics
import numpy as np
import sys

from torch.utils.data import Dataset, random_split
from config import device

import CNNmodel

torch.manual_seed(1)

# This class allows train/test split with different transforms
class DatasetFromSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Test network on validation set, if it exists.
def test_network(net,testloader,print_confusion=False):
    net.eval()
    total_images = 0
    total_correct = 0
    conf_matrix = np.zeros((8,8))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            conf_matrix = conf_matrix + metrics.confusion_matrix(
                labels.cpu(),predicted.cpu(),labels=[0,1,2,3,4,5,6,7])

    model_accuracy = total_correct / total_images * 100
    print(', {0} test {1:.2f}%'.format(total_images,model_accuracy))
    if print_confusion:
        np.set_printoptions(precision=2, suppress=True)
        print(conf_matrix)
    net.train()

def main():
    print("Using device: {}"
          "\n".format(str(device)))
    ########################################################################
    #######                      Loading Data                        #######
    ########################################################################
    data = torchvision.datasets.ImageFolder(root=CNNmodel.dataset)
    
    if CNNmodel.train_val_split == 1:
        # Train on the entire dataset
        data = torchvision.datasets.ImageFolder(root=CNNmodel.dataset,
                            transform=CNNmodel.transform('train'))
        trainloader = torch.utils.data.DataLoader(data,
                            batch_size=CNNmodel.batch_size, shuffle=True);
    else:
        # Split the dataset into trainset and testset
        data = torchvision.datasets.ImageFolder(root=CNNmodel.dataset)
        data.len=len(data)
        train_len = int((CNNmodel.train_val_split)*data.len)
        test_len = data.len - train_len
        train_subset, test_subset = random_split(data, [train_len, test_len])
        trainset = DatasetFromSubset(
            train_subset, transform=CNNmodel.transform('train'))
        testset = DatasetFromSubset(
            test_subset, transform=CNNmodel.transform('test'))

        trainloader = torch.utils.data.DataLoader(trainset,
                            batch_size=CNNmodel.batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, 
                            batch_size=CNNmodel.batch_size, shuffle=False)

    # Get model, loss criterion and optimizer from CNNmodel
    net = CNNmodel.net.to(device)
    criterion = CNNmodel.loss_func
    optimizer = CNNmodel.optimizer
    # get weight initialization and lr scheduler, if appropriate
    weights_init = CNNmodel.weights_init
    scheduler = CNNmodel.scheduler

    # apply custom weight initialization, if it exists
    net.apply(weights_init)
    
    ########################################################################
    #######                        Training                          #######
    ########################################################################
    print("Start training...")
    for epoch in range(1,CNNmodel.epochs+1):
        total_loss = 0
        total_images = 0
        total_correct = 0

        for batch in trainloader:           # Load batch
            images, labels = batch 
            images = images.to(device)
            labels = labels.to(device)

            preds = net(images)             # Process batch
            
            loss = criterion(preds, labels) # Calculate loss

            optimizer.zero_grad()
            loss.backward()                 # Calculate gradients
            optimizer.step()                # Update weights

            output = preds.argmax(dim=1)

            total_loss += loss.item()
            total_images += labels.size(0)
            total_correct += output.eq(labels).sum().item()

        # apply lr schedule, if it exists
        if scheduler is not None:
            scheduler.step()
            
        model_accuracy = total_correct / total_images * 100
        print('ep {0}, loss: {1:.2f}, {2} train {3:.2f}%'.format(
               epoch, total_loss, total_images, model_accuracy), end='')

        if CNNmodel.train_val_split < 1:
            test_network(net,testloader,
                         print_confusion=(epoch % 10 == 0))
        else:
            print()

        if epoch % 10 == 0:
            torch.save(net.state_dict(),'checkModel.pth')
            print("   Model saved to checkModel.pth")        

        sys.stdout.flush()

    torch.save(net.state_dict(),'savedModel.pth')
    print("   Model saved to savedModel.pth")
        
if __name__ == '__main__':
    main()
