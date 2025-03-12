import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
from dataset.generate_data import data_loader
from model.network import ExampleNetwork
from dataset.generate_data import mnist_dataloader


criterion = nn.CrossEntropyLoss()
net = ExampleNetwork(in_feat=3, out_feat=5, D=2)
print(net)

# a data loader must return a tuple of coords, features, and labels.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = net.to(device)
optimizer = SGD(net.parameters(), lr=1e-1)



# # look at the misnt dataset in image form
# data_loader = mnist_dataloader()

# #print the first 5 images in the dataset
# for im in data_loader:
#     plt.imshow(im[0][0][0])
#     plt.savefig('mnist.png')
#     plt.show()

for i in range(10):
        optimizer.zero_grad()

        # Get new data
        coords, feat, label = data_loader()
        input = ME.SparseTensor(feat, coords, device=device)
        label = label.to(device)

        # Forward
        output = net(input)

        # Loss
        loss = criterion(output.F, label)
        print('Iteration: ', i, ', Loss: ', loss.item())

        # Gradient
        loss.backward()
        optimizer.step()


# # Saving and loading a network
# torch.save(net.state_dict(), 'test.pth')
# net.load_state_dict(torch.load('test.pth'))