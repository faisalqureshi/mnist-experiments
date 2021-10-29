import math
import os
import numpy as np
import torch
import torchvision as tv

class Digits(torch.utils.data.Dataset):
    def __init__(self, datafolder, digits):
        """
        A little wrapper around MNIST dataset.
        
        mnist_digits: object returned from torchvision.datasets.MNIST
        digits: 'all' or a list of digits [1, 2, 3, ...] when interested in a subset of MNIST data
        returns:
            when used with dataloaders, returns items of the form {'sample': 1x28x28, 'label': digit}
        """
        
        try:
            self.samples = torch.load(os.path.join(datafolder, 'mnist_samples.pt'))
            self.labels = torch.load(os.path.join(datafolder, 'mnist_labels.pt'))
        
            if not self.samples.shape == (60000, 1, 28, 28) or not self.labels.shape == (60000, 1):
                raise
        except:
            # We didn't find precompiled data
            my_transforms = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.ConvertImageDtype(torch.float),
                tv.transforms.Normalize((0.5,), (0.5,))
            ])

            mnist_digits = tv.datasets.MNIST(datafolder, transform=my_transforms, download=True)
            n = len(mnist_digits)
            assert(n == 60000)
            
            self.samples = torch.empty(n, 1, 28, 28)
            self.labels = torch.empty(n, 1)
        
            for i in range(n):
                self.samples[i,...] = mnist_digits[i][0]
                self.labels[i,] = mnist_digits[i][1]
            
            try:
                torch.save(self.samples, os.path.join(datafolder, 'mnist_samples.pt'))
                torch.save(self.labels, os.path.join(datafolder, 'mnist_labels.pt'))
            except:
                print(f'Error saving compiled data to {datafolder}')
        
        if digits == 'all':
            self.n = self.samples.shape[0]
        else:
            idxs = torch.full([self.samples.shape[0]], False)
            for i in digits:
                idxs = torch.logical_or(idxs, self.labels[:,0] == i)
            self.labels = self.labels[idxs]
            self.samples = self.samples[idxs]
        
    def __len__(self):
        return self.samples.shape[0]
    
    def __getitem__(self, idx):
        return {'sample': self.samples[idx], 'label': self.labels[idx]}

def set_mnist_data(datafolder='./datasets', digits= 'all'):
    """
    digits: 'all' or a list of digits [1, 2, 3, ...] when interested in a subset of MNIST data
    """
    return Digits(datafolder, digits)

def view(dataset, index=-1):
    if index < 0 or index >= len(dataset):
        i = np.random.choice(len(dataset))
    else:
        i = index
    
    img = dataset[i]['sample'].squeeze()
    label = dataset[i]['label'].squeeze().numpy().astype(np.int32)
    return img, label

def split_training_validation_test(dataset, n_train, n_validation, n_test, shuffle=False):
    """
    n_train, n_validation, n_test must be between 0.0 and 1.0 and these must
    sum to 1.0.
    """
    assert(n_train + n_validation + n_test == 1.0)
    
    n = len(dataset)
    idx = np.arange(n)
    if shuffle:
        np.shuffle(idx)


    n_train = int(math.floor(n * n_train))
    n_validation = int(math.floor(n * n_validation))

    training_dataset = torch.utils.data.Subset(dataset, idx[:n_train])
    validation_dataset = torch.utils.data.Subset(dataset, idx[n_train:n_train+n_validation])
    test_dataset = torch.utils.data.Subset(dataset, idx[n_train+n_validation:])
    n_test = len(test_dataset)

    print('======================================')
    print('Datasets:')
    print(f'  Total samples: {n}')
    print(f'  Training samples: {n_train}')
    print(f'  Validation samples: {n_validation}')
    print(f'  Test samples: {n_test}')
    print('======================================')

    return training_dataset, validation_dataset, test_dataset

def set_dataloaders(training_dataset, validation_dataset, test_dataset, batch_size, shuffle=False):
    training_dataloader = torch.utils.data.DataLoader(training_dataset, 
                                                      batch_size=batch_size,
                                                      shuffle=shuffle)

    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle)
    
    print('======================================')
    print('Dataloaders:')
    print(f'  Batch size: {batch_size}')
    print(f'  Shuffle: {shuffle}')
    print(f'  Training batches: {len(training_dataloader)}')
    print(f'  Validation batches: {len(validation_dataloader)}')
    print(f'  Test batches: {len(test_dataloader)}')
    print('======================================')

    return training_dataloader, validation_dataloader, test_dataloader 