import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from config import Config

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)
])

train_dataset = datasets.FashionMNIST(download=True, 
                      root='./data', 
                      train=True, 
                      transform=data_transforms)
test_dataset = datasets.FashionMNIST(download=True, 
                      root='./data', 
                      train=False, 
                      transform=data_transforms)

trainloader = DataLoader(train_dataset, 
                         batch_size=Config.batch_size,
                         num_workers=4,
                         shuffle=True)
testloader = DataLoader(test_dataset, 
                         batch_size=Config.batch_size,
                         num_workers=4,
                         shuffle=True)

