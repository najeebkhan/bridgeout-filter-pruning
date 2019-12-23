import torchvision.datasets   
import torchvision.transforms
import numpy as np
import torch


def get_data_loaders():     
    transformImg = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    valid = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformImg)
    test = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformImg)  
    
    # create training and validation set indexes (80-20 split)
    idx = list(range(len(train)))
    np.random.seed(1009)
    np.random.shuffle(idx)          
    train_idx = idx[ : int(0.8 * len(idx))]       
    valid_idx = idx[int(0.8 * len(idx)) : ]
    
    
    # generate training and validation set samples
    train_set = torch.utils.data.sampler.SubsetRandomSampler(train_idx)    
    valid_set = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)  
    
    # Load training and validation data based on above samples
    # Size of an individual batch during training and validation is 30
    # Both training and validation datasets are shuffled at every epoch by 'SubsetRandomSampler()'. Test set is not shuffled.
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, sampler=train_set, num_workers=4)  
    valid_loader = torch.utils.data.DataLoader(train, batch_size=32, sampler=valid_set, num_workers=4)    
    test_loader = torch.utils.data.DataLoader(test, num_workers=4)
    return train_loader, valid_loader, test_loader


 