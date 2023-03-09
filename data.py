import torch
import torchvision
import torchvision.transforms as trans
import time

def load_data():
    '''return train set loader & test set loader'''
    tic = time.time()
    transform = trans.Compose([trans.ToTensor(),trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)
    toc = time.time()
    print('load data in %.3f' % (toc - tic))
    return trainloader, testloader

def get_classes(label):
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes[label]