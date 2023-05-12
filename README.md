# FashionMnistPytorch

A Neural Network trained on Fashion MNIST data using PyTorch.
All the training logs are tracked on Weights & Biases. Refer the [Quickstart guide](https://wandb.ai/quickstart) to set it up.

## Getting Started
1. Spin up a Docker continer and install the dependencies in `requirements.txt`
2. Open the terminal and run,
```
> python3 train.py
```
3. Experiment with hyperparameters in `train.py` file and view your saved models in `/models` folder or `WandB dashboard`

Future Additions I'm working on:
1. WandB artifact tracking for saving my models
2. Performing Hyperparameter Sweeps
3. Playing with various dataset cross-folds to make my model more robust