import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from tqdm import tqdm
import numpy as np


def main():
    model = Net()
    t = Trainer(model, num_epochs=20)
    while t.next_epoch():
        t.train()
        t.valid()


class Net:
    pass


class Trainer:  # GAN은 상속받아서 쓰셈
    # TODO: timer, scheduler
    def __init__(self, model, num_epochs, loss_fn):
        self.num_epochs = num_epochs
        self.epoch = -1  # 0-indexed which schedulers can easily handle
        self.model = model
        self.criterion = loss_fn
        self.train_losses = {0: []}
        self.valid_losses = {0: []}
        self.train_loader, self.valid_loader = get_dataloaders()

    def next_epoch(self):
        self.epoch += 1
        return self.epoch < self.num_epochs

    def train(self):
        self.train_losses[self.epoch] = []
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.train_loader)):
            self.__train_step(inputs, labels)

    @torch.no_grad()
    def valid(self):  # validate
        self.valid_losses[self.epoch] = []
        for batch_idx, (inputs, labels) in enumerate(tqdm(self.valid_loader)):
            self.__valid_step(inputs, labels)
        self.__log()
    
    def __train_step(self, inputs, labels):
        self.__init_step(inputs, labels)
        self.__forward()
        self.__calc_loss()
        self.__backward()

    def __valid_step(self, inputs, labels):
        self.__init_step(inputs, labels)
        self.__forward()
        self.__calc_loss(True)
    
    def __log(self):
        pass

    def __init_step(self, inputs, labels):
        self.inputs, self.labels = inputs.to(self.device), labels.to(self.device)
        self.optim.zero_grad(set_to_none=True)
    
    def __forward(self):
        self.outputs = self.model(self.inputs)

    def __calc_loss(self, valid_mode=False):
        self.loss = self.criterion(self.outputs, self.labels)
        _loss = self.loss.detach().cpu().item()
        [self.train_losses, self.valid_losses][valid_mode][self.epoch].append(_loss)

    def __backward(self):
        self.loss.backward()
        self.optim.step()


def get_dataloaders():
    data_train, data_valid = get_datasets()
    train_loader = torch.utils.data.DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    valid_loader = torch.utils.data.DataLoader(data_valid, batch_size=256, shuffle=False, num_workers=8)
    return train_loader, valid_loader


def get_datasets():
    data_train = torchvision.datasets.MNIST(
        root='~/Datasets',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32, 32)),
        ])
    )
    data_valid = torchvision.datasets.MNIST(
        root='~/Datasets',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Resize((32, 32)),
        ])
    )
    return data_train, data_valid


if __name__ == '__main__':
    main()
