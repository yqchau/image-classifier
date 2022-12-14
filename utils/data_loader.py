import hydra
import torch
import torchvision


def get_data_loaders(
    train_dataset_path,
    val_dataset_path,
    train_transform,
    val_transform,
    batch_size,
    num_workers,
):

    train_transform = hydra.utils.instantiate(train_transform)
    val_transform = hydra.utils.instantiate(val_transform)

    train_dataset = torchvision.datasets.ImageFolder(
        train_dataset_path, train_transform
    )
    val_dataset = torchvision.datasets.ImageFolder(val_dataset_path, val_transform)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_data_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_data_loader, val_data_loader
