import torch
import torch.nn as nn
import torchmetrics
import wandb


def train_model(model, dataloader, loss_fn, optimizer):
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

def log_img_table(imgs, preds, labels, probs):
    num_classes = probs.size()[-1]
    table = wandb.Table(columns=["image", "pred", "target"]+[f"score_{i}" for i in range(num_classes)])
    for img, pred, targ, prob in zip(imgs, preds, labels, probs):
        table.add_data(wandb.Image(img[0].numpy()*255), pred, targ, *probs.numpy())
    wandb.log({
        "predictions_table": table
    }, commit=False)

def test_model(model, dataloader, loss_fn, num_classes, log_imgs=False):
    """Computes loss and accuracy of the model on the dataset. It also logs one prediction batch to wandb table if `log_imgs` is set to TRUE

    Args:
        model
        dataloader
        loss_fn
        num_classes
        log_imgs

    Returns:
        epoch_loss (float): Accumulated value of losses across all batches
        epoch_acc (float): Prediction accuracy of the model
    """
    model.eval()
    acc_metric = torchmetrics.classification.MulticlassAccuracy(
        num_classes, average=None
    )
    epoch_loss = 0
    with torch.inference_mode():
        for batch_num, batch in enumerate(dataloader):
            X, y = batch
            # forward prop
            with torch.no_grad():
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                epoch_loss += loss.item()
                batch_acc = acc_metric(y_hat, y)
                if log_imgs and batch_num == 0:
                    _, preds = torch.max(y_hat.data, 1)
                    log_img_table(
                        X,
                        preds,
                        y,
                        y_hat.softmax(dim=1)
                    )

    epoch_acc = acc_metric.compute()
    return epoch_loss, torch.mean(epoch_acc)



# from model import FeedForwardNeuralNet
# from data import get_loader

# def test_test_epoch_function() -> None:
#     loader = get_loader(True, 32)
#     model = FeedForwardNeuralNet((28,28),10)
#     loss_fn = nn.CrossEntropyLoss()

#     print(test_model(model=model,
#                       dataloader=loader[1],
#                       loss_fn=loss_fn,
#                       num_classes=10, log_imgs=True))
# test_test_epoch_function()
