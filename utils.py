import torch
import torchmetrics

from config import Config

def train_epoch(model, dataloader, loss_fn, optimizer):
    """Trains the model for 1 epoch across all the dataset batches

    Args:
        model 
        dataloader 
        loss_fn 
        optimizer 

    Returns:
        epoch_loss (float): Accumulated value of losses across all batches
    """
    dataset_size = len(dataloader)
    epoch_loss = 0
    for batch_num, batch in enumerate(dataloader):
        X, y = batch
        # forward prop
        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        epoch_loss += loss.item()

        # backward prop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return epoch_loss


def test_epoch(model, dataloader, loss_fn):
    """Computes loss and accuracy of the model on the dataset

    Args:
        model 
        dataloader
        loss_fn 

    Returns:
        epoch_loss (float): Accumulated value of losses across all batches
        epoch_acc (float): Prediction accuracy of the model
    """
    dataset_size = len(dataloader)
    epoch_loss = 0
    acc_metric = torchmetrics.Accuracy(task = 'multiclass', num_classes=Config.num_classes)
    for batch_num, batch in enumerate(dataloader):
        X, y = batch
        # forward prop
        with torch.no_grad():
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            epoch_loss += loss.item()
            batch_acc = acc_metric(y_hat, y)

    epoch_acc = acc_metric.compute()
    return epoch_loss, epoch_acc