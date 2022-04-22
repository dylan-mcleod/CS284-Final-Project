import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

batch_size = 512
epochs = 20
learning_rate = 1e-3

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class AE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        self.model = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2D(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.ReLu(),
            nn.BatchNorm2d(hidden_dim),
            *[nn.Sequential(nn.Conv2D(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=5,stride=2), nn.ReLu(), nn.BatchNorm2d(hidden_dim)) for _ in range(layers)],
            nn.Conv2D(in_channels=hidden_dim,out_channels=out_dim,kernel_size=1))

    def forward(self, features):
        return model.forward(features)
        
train_data = torchvision.datasets.ImageFolder(root="Data/RogalandImages/elevation")
train_loader = data.DataLoader(train_data, batch_size=40, shuffle=True,  num_workers=4)
test_data = torchvision.datasets.ImageFolder(root="Data/RogalandImages/elevation")
test_loader  = data.DataLoader(test_data, batch_size=40, shuffle=True, num_workers=4)

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create a model from `AE` autoencoder class
# load it to the specified device, either gpu or cpu
model = AE(1, 64, 1, 5).to(device)


optimizer = torch.optim.SGD(model.model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.MSELoss()
for epoch in range(epochs):
    loss = 0
    for batch_features, _ in train_loader:
        # reset the gradients back to zero
        optimizer.zero_grad()
        
        # compute reconstructions
        outputs = model(batch_features)
        
        # compute training reconstruction loss
        train_loss = criterion(outputs, batch_features)
        
        # compute accumulated gradients
        train_loss.backward()
        
        # perform parameter update based on current gradients
        optimizer.step()
        
        # add the mini-batch training loss to epoch loss
        loss += train_loss.item()
    
    # compute the epoch training loss
    loss = loss / len(train_loader)
    
    # display the epoch training loss
    print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
    

test_examples = None

with torch.no_grad():
    for batch_features in test_loader:
        batch_features = batch_features[0]
        test_examples = batch_features.to(device)
        reconstruction = model(test_examples)        
        number = 10
        plt.figure(figsize=(20, 4))
        for index in range(number):
            # display original
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(128, 128))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # display reconstruction
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(128, 128))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
        break
        