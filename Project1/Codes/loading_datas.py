import torch
from torchvision import datasets

import os


def mnist_to_pairs(nb, train, target):
    input = torch.functional.F.avg_pool2d(train, kernel_size = 2)
    a = torch.randperm(input.size(0)) #permutation of indices
    nbTotalPairs = input.size(0)//2 # create maximum number of pairs, we take a specific number at the end
    a = a[:2 * nbTotalPairs].view(nbTotalPairs, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] >= classes[:, 1]).long()
    
    sameClasses = (classes[:, 0] != classes[:, 1]).nonzero().flatten() # find the indices where the classes are not the same
    # since we want only different classes in our pairs
    pairs = torch.index_select(input, 0, sameClasses)[:nb]
    classes = torch.index_select(classes, 0, sameClasses)[:nb]
    target = torch.index_select(target, 0, sameClasses)[:nb]

    return pairs, target, classes


######################################################################


def generate_pair_sets(nb):
    
    data_dir = os.environ.get('PYTORCH_DATA_DIR')
    if data_dir is None:
        data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + mnist_to_pairs(nb, test_input, test_target)

######################################################################