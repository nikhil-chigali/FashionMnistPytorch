import torch.nn as nn
import torch.optim as optim
from model import MyModel
import data
from tqdm import tqdm
from utils import train_epoch, test_epoch
from config import Config

if __name__ == '__main__':
    optimizer = optim.SGD(lr=Config.lr)
    loss_fn = nn.CrossEntropyLoss()
    model = MyModel()

    # Training
    for e in tqdm(range(Config.epochs), desc='Training'):
        loss = train_epoch(model, 
                    dataloader=data.trainloader, 
                    optimizer=optimizer, 
                    loss_fn=loss_fn)
        print(f"Epoch {e+1}/{Config.epochs} | Loss: {loss}")
