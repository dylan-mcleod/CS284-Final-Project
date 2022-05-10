import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io, transform
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import PIL

# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

class AE(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers):
        nn.Module.__init__(self)
        self.xmodel = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_dim),
            *[nn.Sequential(nn.Conv2d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(hidden_dim)) for _ in range(layers)],
            nn.Conv2d(in_channels=hidden_dim,out_channels=out_dim,kernel_size=1))

    def forward(self, features):
        return self.xmodel.forward(features)
        
class LandHeightsDataset(torch.utils.data.Dataset):
    """ Segmented Land Heights dataset.

        Sample format:
            {
                class: integer (e.g. rogaland = 0, uganda = 1)
                cover: image
                elevation: image
                rgb: image
            }
    """

    def __init__(self, root_dir, subfolders, transform=None):
        self.root_dir = root_dir
        self.subfolders = []
        self.subnames = subfolders[:]
        self.pathlens = []
        self.len_calc = 0
        self.imagetypes = ("cover", "elevation")
        for f in subfolders:
            basepath = os.path.join(self.root_dir, f)
            pth = [os.path.join(basepath, x) for x in self.imagetypes]
            self.subfolders.append(pth)
            l = len(os.listdir(pth[0]))
            self.pathlens.append(l)
            self.len_calc += l
        self.transform = transform

    def __len__(self):
        return self.len_calc

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find corresponding dataset subfolder (e.g. rogaland, uganda)
        sfolder, sname, si = None,None,0
        _idx = idx
        for i, l in enumerate(self.pathlens):
            if _idx < l:
                sfolder, sname, i = self.subfolders[i], self.subnames[i], i
            else:
                _idx -= l

        sample = {'class':si}

        # Get images from files -- loop through image types (e.g. cover, elevation, rgb)
        imgname = sname + str(_idx) + '.png'
        for i,x in enumerate(self.imagetypes):
            img_name = os.path.join(sfolder[i], imgname)
            image = io.imread(img_name)
            sample[x] = torchvision.transforms.ToTensor()(image)[:1]

        if self.transform:
            sample = self.transform(sample)

        return sample

if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    batch_size = 512
    epochs = 20
    learning_rate = 1e-3
    #  use gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_data = LandHeightsDataset("Data", ["Uganda"])
    train_loader = data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=10)
    #test_data = LandHeightsDataset("Data", ["rogaland"], transform)
    #test_loader  = data.DataLoader(test_data, batch_size=40, shuffle=True, num_workers=4)


    # create a model from `AE` autoencoder class
    # load it to the specified device, either gpu or cpu
    model = AE(1, 64, 1, 5).to(device)


    optimizer = torch.optim.SGD(model.xmodel.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        loss = 0
        for i, batch in enumerate(train_loader):
            # reset the gradients back to zero
            optimizer.zero_grad()
            
            # compute reconstructions
            outputs = model(batch['elevation'].to(device))
            
            # compute training reconstruction loss
            train_loss = criterion(outputs, batch['elevation'].to(device))
            
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
        
    '''
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
    '''
        