import torch
import MinkowskiEngine as ME
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np


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
):
    if seed >= 0:
        torch.manual_seed(seed)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    for data, label in dataloader:
        coords = []
        for i, row in enumerate(data):
            for j, col in enumerate(row):
                if col != 0:
                    coords.append([i, j])
        coords = np.array(coords)
        coords = ME.utils.batched_coordinates([coords])
        feats = data.view(data.shape[0], -1).to(dtype)
        label = label.to(dtype)
        return coords, feats, label



