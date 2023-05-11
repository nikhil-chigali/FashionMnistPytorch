import torch.nn as nn
import torch.optim as optim
from model import FeedForwardNeuralNet
from data import get_loader
from train_utils import train_model, test_model, validate_model
from tqdm import tqdm
import wandb


def initialize():
    global config, dataloaders, loss_fn
    global model, optimizer

    # Initializing WandB project
    wandb.init(
        project="fashionmnist-pytorch",
        config={
            "epochs": 20,
            "batch_size": 32,
            "lr": 0.01,
            "num_classes": 10,
            "img_size": (28, 28),
            "dataset": "FashionMNIST",
            "model": "FeedForwardNeuralNetwork",
        },
    )
    config = wandb.config

    dataloaders = get_loader(is_train=True, batch_size=config.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    model = FeedForwardNeuralNet(config.img_size, config.num_classes)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)


if __name__ == "__main__":
    initialize()

    train_loader, val_loader = dataloaders

    # Training
    for e in tqdm(range(config.epochs), desc="Training"):
        loss = train_model(
            model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn
        )
        wandb.log({"train": {"loss": loss}}, step=e)

        print(f"Epoch {e+1}/{config.epochs} | Training Loss: {loss:.03f}", end=" ")
        if (e + 1) % 5 == 0 or (e + 1) == config.epochs:
            val_loss, val_acc = test_model(
                model, val_loader, loss_fn, config.num_classes, log_imgs=True
            )
            wandb.log({"val": {"loss": val_loss, "acc": val_acc}}, step=e)
            print(f"| Val Loss: {val_loss:.03f} ; Val Acc: {val_acc*100:.03f}%")
        else:
            print("")