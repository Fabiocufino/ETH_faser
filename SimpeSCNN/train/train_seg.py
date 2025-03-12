import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import MinkowskiEngine as ME
from dataset.generate_data import data_loader, mnist_dataloader
from model.network import SCNN_MNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

D = 3  # Dimension for MinkowskiEngine
net = SCNN_MNIST(D).to(device)
print(net)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

# Debug: Before Processing
coords, feats, labels = mnist_dataloader()
print("Before processing:")
print(coords)
print(f"Coords shape: {coords.shape}")  # (N, 3)
print(feats)
print(f"Feats shape: {feats.shape}")    # (N, 1)
print(f"Expected D: {D}")

for i in range(10):
    optimizer.zero_grad()

    # Get new data
    coords, feats, labels = mnist_dataloader()

    # Generate batch indices correctly
    batch_size = coords.shape[0]
    batch_indices = torch.zeros((batch_size, 1), dtype=torch.int32)

    # Ensure (N, D+1) shape
    coords = torch.cat([batch_indices, coords], dim=1)

    # Ensure (N, C) shape for features
    feats = feats.view(coords.shape[0], -1)  # Match coords

    # Create SparseTensor
    input_tensor = ME.SparseTensor(features=feats, coordinates=coords, device=device)

    # Forward Pass
    output = net(input_tensor)

    print(f"Output shape: {output.F.shape}")

    # Compute Loss
    labels = labels[:1]
    labels = labels.view(-1).long().to(device)  # Ensure correct shape

    loss = criterion(output.F, labels)

    print(f"Iteration {i}: Loss {loss.item()}")

    # Backpropagation
    loss.backward()
    optimizer.step()

# Save & Load Model
torch.save(net.state_dict(), "test.pth")
net.load_state_dict(torch.load("test.pth"))
