import torch.nn as nn
import torch.optim as optim
from model import MyModel
from data import trainloader
from tqdm import tqdm
from utils import train_epoch, test_epoch
from config import Config

if __name__ == '__main__':
    loss_fn = nn.CrossEntropyLoss()
    model = MyModel()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr)

    # Training
    for e in tqdm(range(Config.epochs), desc='Training'):
        loss = train_epoch(model, 
                    dataloader=trainloader, 
                    optimizer=optimizer, 
                    loss_fn=loss_fn)
        print(f"Epoch {e+1}/{Config.epochs} | Loss: {loss:.03f}")
    
    # Testing Accuracy and Loss
    loss, acc = test_epoch(model, trainloader, loss_fn)
    print(f"Loss on Train-set: {loss:.03f} \nAccuracy on Train-set: {acc*100:.03f}%")
