import torch
import MinkowskiEngine as ME
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt

def get_coords(data):
    coords = []
    for i, row in enumerate(data):
        for j, col in enumerate(row):
            if col != " ":
                coords.append([i, j])
    return np.array(coords)


def data_loader(
    nchannel=3,
    max_label=5,
    is_classification=True,
    seed=-1,
    batch_size=2,
    dtype=torch.float32,
):
    if seed >= 0:
        torch.manual_seed(seed)

    data = ["   X   ",
            "  X X  ",
            " XXXXX "]

    # Generate coordinates
    coords = [get_coords(data) for i in range(batch_size)]
    coords = ME.utils.batched_coordinates(coords)

    # features and labels
    N = len(coords)
    feats = torch.arange(N * nchannel).view(N, nchannel).to(dtype)
    label = (torch.rand(batch_size if is_classification else N) * max_label).long()
    return coords, feats, label



#def dataloader downloading the minst dataset for classification
def mnist_dataloader(
    batch_size=2,
    num_workers=2,
    is_classification=True,
    seed=-1,
    dtype=torch.float32,
    max_label=10  # Maximum label value for random shuffling
):
    if seed >= 0:
        torch.manual_seed(seed)

    # Define transformation (convert to tensor only)
    transform = transforms.Compose([transforms.ToTensor()])

    # Load dataset
    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    def get_coords_and_feats(img):
        """ Extracts nonzero pixel coordinates and corresponding features from an image. """
        nonzero_indices = (img[0] != 0).nonzero(as_tuple=False)  # Get coordinates
        features = img[0][nonzero_indices[:, 0], nonzero_indices[:, 1]].unsqueeze(1)  # Get pixel values
        return nonzero_indices, features

    # Process each batch
    for data, labels in dataloader:
        coords_list = []
        feats_list = []

        for i in range(batch_size):
            coords, feats = get_coords_and_feats(data[i])  # Extract per-image
            coords_list.append(torch.cat((torch.full((coords.shape[0], 1), i, dtype=torch.int32), coords), dim=1))
            feats_list.append(feats)

        # Convert lists to tensors
        coords = torch.cat(coords_list, dim=0)  # Batched coordinates
        feats = torch.cat(feats_list, dim=0).to(dtype)  # Matching features

        # Generate shuffled labels
        shuffled_labels = torch.randint(0, max_label, (batch_size if is_classification else coords.shape[0],))
        print("All the shapes", coords.shape, feats.shape, shuffled_labels.shape)
        return coords, feats, shuffled_labels  # Return first batch only

