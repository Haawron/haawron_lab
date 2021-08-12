import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import torchvision.transforms as TF

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def get_datasets():
    mnist_train = torchvision.datasets.MNIST(
        root='~/Datasets',
        train=True,
        download=True,
        transform=TF.Compose([
            TF.ToTensor(),
            TF.Normalize((0.1307,), (0.3081,)),
            TF.Resize((32, 32)),
        ])
    )

    mnist_test = torchvision.datasets.MNIST(
        root='~/Datasets',
        train=False,
        download=True,
        transform=TF.Compose([
            TF.ToTensor(),
            TF.Normalize((0.1307,), (0.3081,)),
            TF.Resize((32, 32)),
        ])
    )
    return mnist_train, mnist_test


def get_dataloaders():
    mnist_train, mnist_test = get_datasets()
    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=8)
    return train_loader, test_loader


class Net(nn.Sequential):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.AvgPool2d(4, 4),
            nn.Conv2d(128, 10, 1, 1, 0),
            nn.Softmax(dim=1)
        )


def KL(p, q):
    """N x C x H x W = -1 x 10 x 1 x 1"""
    eps = 1e-5
    loss = p * (torch.log(p+eps) - torch.log(q+eps))
    loss = loss.sum(dim=1).mean()
    return loss


def get_loss(outputs, labels, cooling_factor=5., forward=True):
    one_hot = F.one_hot(labels, num_classes=10)
    if forward:
        loss = KL(one_hot, torch.squeeze(outputs))  # Forward
    else:
        soft_one_hot = F.softmax(cooling_factor*one_hot, dim=1)
        loss = KL(torch.squeeze(outputs), soft_one_hot)  # Backward
    return loss


def test_loss():
    y = F.one_hot(torch.tensor((1,)), num_classes=5)
    y_ = torch.rand(1, 5)
    y_ /= y_.sum(dim=1, keepdims=True)
    print('\ntest')
    print(f'inputs\n\ty : {y}\n\ty_: {y_}')
    print('forward ', KL(y, y_).item())
    print('backward', KL(y_, y).item(), '\n')


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np2torch = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)
    torch2np = lambda tensor: tensor.detach().cpu().numpy() 

    train_loader, test_loader = get_dataloaders()
    net = Net().to(device)
    print(net)
    test_loss()

    optim = torch.optim.SGD(net.parameters(), lr=.02, momentum=.9)
    losses = []
    for epoch in range(1, 10+1):
        epoch_losses = []
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            optim.zero_grad(set_to_none=True) 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = get_loss(outputs, labels, cooling_factor=4.75, forward=False)
            epoch_losses += [loss.detach().cpu().item()]
            loss.backward()
            optim.step()
        losses += [epoch_losses]
        print(f'Epoch {epoch:2d}  loss: {np.mean(epoch_losses)}') 

    conf = np.zeros((10, 10), dtype=np.int32) 
    for inputs, labels in tqdm(test_loader):
        with torch.no_grad():
            out = net(inputs.to(device)).squeeze()
        out = torch2np(out).argmax(axis=1).reshape(-1) 
        for out_, label in zip(out, labels):
            conf[out_, label] += 1
    print(f'acc: {np.diag(conf).sum() / conf.sum():.2%}') 
    print(conf)


if __name__=='__main__':
    main()
