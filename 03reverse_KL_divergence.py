import torch
import torch.nn as nn
import torchvision
from torch.nn import functional as F
import torchvision.transforms as TF

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class Net(nn.Sequential):
    def __init__(self):
        super().__init__()
        feats = [1, 64, 256, 256, 512, 10]
        for i in range(5):
            self.add_module(
                f'conv{i+1}',
                self.build_conv(feats[i], feats[i+1], i!=4)
            )
    def build_conv(self, c_in, c_out, act=True):
        m = nn.ModuleList()
        m += [
            nn.Conv2d(c_in, c_in, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in, c_out, 3, 2, 1),]
        if act:
            m += [nn.ReLU(inplace=True)]
        else:
            m += [nn.Softmax(dim=1)]
        return nn.Sequential(*m)


def KL(p, q):
    """N x C x H x W = -1 x 10 x 1 x 1"""
    eps = 1e-5
    loss = p * (torch.log(p+eps) - torch.log(q+eps))
    loss = loss.sum(dim=1).mean()
    return loss


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    np2torch = lambda arr: torch.tensor(arr, dtype=torch.float32, device=device)
    torch2np = lambda tensor: tensor.detach().cpu().numpy() 

    net = Net().to(device)

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

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=128, shuffle=True, num_workers=8)

    print(net)
    print(len(train_loader), train_loader.batch_size)

    y = torch.rand(12, 10, 1, 1)
    y /= y.sum(dim=1, keepdims=True)
    y_ = torch.rand(12, 10)
    y_ /= y_.sum(dim=1, keepdims=True) 
    print(KL(y, y_), KL(y, y), KL(y_, y), y.sum(dim=1).squeeze()) 

    optim = torch.optim.SGD(net.parameters(), lr=.002)
    losses = []
    for epoch in range(1, 5+1):
        epoch_losses = []
        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optim.zero_grad(set_to_none=True) 
            outputs = net(inputs)
            loss = KL(F.one_hot(labels), outputs)  # Forward
            #loss = KL(outputs, F.one_hot(labels))  # Backward
            epoch_losses += [loss.cpu().item()]
            loss.backward()
            optim.step()
        losses += [epoch_losses]
        print(f'Epoch {epoch:2d}  loss: {np.mean(epoch_losses)}') 

    mnist_test_viz = torchvision.datasets.MNIST(
        root='~/Datasets',
        train=False,
        download=True,
        transform=TF.Compose([
            # TF.ToTensor(),
            # TF.Normalize((0.1307,), (0.3081,)),
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

    conf = np.zeros((10, 10)) 
    for inputs, labels in tqdm(torch.utils.data.DataLoader(mnist_test, batch_size=256, shuffle=False, num_workers=8)):
        with torch.no_grad():
            out = net(inputs.to(device)).squeeze()
        out = torch2np(out).argmax(axis=1).reshape(-1) 
        conf[out, labels] += 1
    print(f'acc: {np.diag(conf).sum() / 1000}') 
    print(conf)


if __name__=='__main__':
    main()
