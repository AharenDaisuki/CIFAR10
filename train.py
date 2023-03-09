import torch.nn as nn
import torch.optim as optim
import time

def train(network, loader):
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    tic = time.time()
    for i in range(2):
        tot_loss = 0.0
        for j, data in enumerate(loader, 0):
            inputs, labels = data
            optimizer.zero_grad() # clear gradient
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            tot_loss += loss.item()
            if j % 2000 == 1999:
                print('[%d %5d] loss: %.3f' % (i+1, j+1, tot_loss/2000))
                tot_loss = 0.0 # reset loss
    toc = time.time()
    print('Training finished in %.3f s' % (toc-tic))
    return
