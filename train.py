import torch.nn as nn
import torch.optim as optim
from model import FeedForwardNeuralNet
from data import get_loader
from tqdm import tqdm
from train_utils import train_epoch, test_epoch
import wandb
from config import Config

def initialize():
    global config, dataloaders, loss_fn
    global model, optimizer

    # Initializing WandB project
    # wandb.init(
    #     project = "fashionmnist-pytorch",
    #     config = {
    #         "epochs": 20,
    #         "batch_size": 32,
    #         "lr": 0.01,
    #         "num_classes": 10,
    #         "img_size": (28,28),
    #         "dataset": "FashionMNIST",
    #         "model": "FeedForwardNeuralNetwork"
    #     }
    # )
    # config = wandb.config

    dataloaders = get_loader(is_train=True, batch_size=Config.batch_size)
    loss_fn = nn.CrossEntropyLoss()
    model = FeedForwardNeuralNet()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr)

if __name__ == '__main__':
    initialize()

    train_loader, val_loader = dataloaders
    # Training
    for e in tqdm(range(Config.epochs), desc='Training'):
        loss = train_epoch(model, 
                    dataloader=train_loader, 
                    optimizer=optimizer, 
                    loss_fn=loss_fn)
        print(f"Epoch {e+1}/{Config.epochs} | Training Loss: {loss:.03f}", end="")
        if (e+1)%5 == 0 or (e+1) == Config.epochs:
            val_loss, val_acc = test_epoch(model, val_loader, loss_fn)
            print(f"Epoch {e+1}/{Config.epochs} | Validation Loss: {loss:.03f}")
        else:
            print('')
