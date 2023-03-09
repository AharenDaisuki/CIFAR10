from data import load_data, get_classes
from network import Network
from train import train

if __name__ == '__main__':
    # data
    trainloader, testloader = load_data()
    # network
    net = Network()
    # training
    train(net, trainloader)
