import torch
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Subset, DataLoader

g = torch.Generator('cpu')
g.manual_seed(seed=42)

data_transforms = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
)

def get_dataset(is_train, data_transforms):
    dataset = datasets.FashionMNIST(
        download=True, root="./data", train=is_train, transform=data_transforms
    )
    return dataset


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_loader(is_train, batch_size):
    """
    Returns Train and Val loaders when training, and
    Test loader when testing

    Args:
        is_train (bool): Training or not
        batch_size (int): Size of the batch

    Returns:
        loader (torch.utils.data.DataLoader): Returns Train, Val and Test dataloaders
    """
    dataset = get_dataset(is_train, data_transforms)
    if is_train:
        dataset_size = len(dataset)
        all_indices = list(range(0, dataset_size))
        random.shuffle(all_indices)
        val_indices = all_indices[: len(dataset) // 4]
        train_indices = all_indices[len(dataset) // 4 :]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_train,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_train,
            worker_init_fn=seed_worker,
            generator=g,
        )
        loader = (train_loader, val_loader)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            shuffle=is_train,
            worker_init_fn=seed_worker,
            generator=g,
        )
    return loader
