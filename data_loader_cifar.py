import os
import pickle

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from torchvision import datasets
from torchvision import transforms


def get_cifar_loader(image_type, opts):

    transform = transforms.Compose([
                    transforms.Scale(opts.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_dataset = datasets.CIFAR10('./', train=True, transform=transform)
    test_dataset = datasets.CIFAR10('./', train=False, transform=transform)

    train_mask = [i for i in range(len(train_dataset)) if train_dataset.train_labels[i] == image_type]
    test_mask = [i for i in range(len(test_dataset)) if test_dataset.test_labels[i] == image_type]

    train_dloader = DataLoader(dataset=train_dataset, batch_size=opts.batch_size,
                               num_workers=opts.num_workers, sampler=SubsetRandomSampler(train_mask))
    test_dloader = DataLoader(dataset=test_dataset, batch_size=opts.batch_size,
                              num_workers=opts.num_workers, sampler=SubsetRandomSampler(test_mask))

    return train_dloader, test_dloader


if __name__ == '__main__':
    from cycle_gan import create_parser

    parser = create_parser()
    opts = parser.parse_args()

    _train, _test = get_cifar_loader(3, opts)
