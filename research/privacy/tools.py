import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

def var(x):
    return Variable(torch.FloatTensor(x))

def column_split(X, selection):
    """
        example: X = array([1,2,3,4]), selection = [0,1]
                 returns array([1,2]),array([3,4])
                 uses deep copy.
    """
    m, n = X.shape
    select = [True if i in selection else False for i in range(n)]
    leave  = [False if i in selection else True for i in range(n)]
    select = X[:,select]
    leave  = X[:,leave]
    return select, leave

def random_column_split(X, size):
    """
        random column split with partition one of given size
        and rest in the second partition
    """
    m, n = X.shape
    return

def classification_traintest(model, Xtr, ytr, Xtst, ytst, epochs = 50):
    optimizer = torch.optim.Adam(model.parameters())
    #criterion = torch.nn.MultiLabelSoftMarginLoss()
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    for epoch in range(epochs):
        x = Variable(torch.FloatTensor(Xtr))
        y = Variable(torch.FloatTensor(ytr))
        output = model(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.data.mean())

    output = model(Variable(torch.FloatTensor(Xtst)))
    loss = criterion(output, Variable(torch.FloatTensor(ytst)))
    return losses, loss

if __name__ == "__main__":
    a = np.array([[1,2,3,4],
                  [2,4,6,8]])
    print(column_split(a,[0,1,2]))
