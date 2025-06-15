import torch
import torch.nn as nn
from torchvision import datasets, transforms

class FMNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def simple_mlp_model():
    model = FMNIST_MLP()
    model.load_state_dict(torch.load("assignment4/fmnist_mlp.pth", map_location='cpu'))
    return model

def fmnist_dataset(spec):
    eps = spec["epsilon"]
    if eps is None:
        eps = 0.1

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X, labels = next(iter(dataloader))

    data_max = torch.clamp(X + eps, -1.0, 1.0)
    data_min = torch.clamp(X - eps, -1.0, 1.0)

    eps_tensor = torch.tensor(eps).reshape(1, 1, 1, 1)

    return X, labels, data_max, data_min, eps_tensor

