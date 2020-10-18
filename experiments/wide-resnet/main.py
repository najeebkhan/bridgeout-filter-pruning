import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

import os
import sys
import time
import argparse
import datetime
import random
import copy

from wide_resnet import Wide_ResNet, conv_init
from torch.autograd import Variable
from pruner import MagnitudePruner
from ray import tune



class Params:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def getArguments():
    parser = argparse.ArgumentParser(description='Wide Resnet Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
    parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
    parser.add_argument('--reg_type', default='bridgeout', type=str, help='regularization')
    parser.add_argument('--depth', default=28, type=int, help='depth of model')
    parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
    parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
    parser.add_argument('--act_norm', default=None, type=float, help='activation norm for sparseout')
    parser.add_argument('--tar_frac', default=0.75, type=float, help='tar_frac')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset = [cifar10/cifar100]')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--testPruningOnly', '-T', action='store_true', help='Test mode with the saved model')
    parser.add_argument('--noAugment', '-n', default=False, help='No data augmentation')
    parser.add_argument('--seed', default=43, type=int, help='seed')

    return parser.parse_args()

    # return Params(
    #         act_norm=None,
    #         dataset='CIFAR10',
    #         depth=34,
    #         dropout=0.3,
    #         lr=0.1,
    #         net_type='wide-resnet',
    #         noAugment=False,
    #         reg_type='bridgeout',
    #         resume=False,
    #         tar_frac=1.0,
    #         testOnly=False,
    #         testPruningOnly=False,
    #         widen_factor=4)

def setSeeds(seed_init):    
    print('seed:',  seed_init)
    random.seed(seed_init)
    torch.manual_seed(seed_init)
    #torch.backends.cudnn.enabled = False
    torch.cuda.manual_seed_all(seed_init)

def write(fname, value):
    with open(fname, 'a') as f:
        f.write(str(value)+"\n")

def getTransforms(args):
    if args.noAugment:
        print('No data augmentation')
        transform_train = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ]) 
    else:        
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
        ]) # meanstd transformation
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
    ])
    return transform_train, transform_test

def getDataset(args):
    transform_train, transform_test = getTransforms(args)
    if args.dataset == 'SVHN':
        dataset = getattr(torchvision.datasets, args.dataset)
        trainset = dataset(root='./data', split='train', download=True, transform=transform_train)
        testset = dataset(root='./data', split='test', download=False, transform=transform_test)
    else:
        dataset = getattr(torchvision.datasets, args.dataset)
        trainset = dataset(root='./data', train=True, download=True, transform=transform_train)
        testset = dataset(root='./data', train=False, download=False, transform=transform_test)
    return trainset, testset

def getNetwork(args):
    regularizer = Params(name=args.reg_type, dropout_rate=args.dropout, q_norm=args.act_norm, target_fraction=args.tar_frac)
        
    num_classes=10
    if args.dataset in ['CIFAR100']:
        num_classes=100
        
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):        
        net = ResNet(args.depth, num_classes, regularizer)
        file_name = 'resnet-'+str(args.depth)+'_'+args.reg_type+'_'+args.dataset+'_p_'+str(args.dropout)+'_q_'+str(args.act_norm)+'_target_'+str(args.tar_frac)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, num_classes, regularizer)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)+'_'+args.reg_type+'_'+args.dataset+'_p_'+str(args.dropout)+'_q_'+str(args.act_norm)+'_target_'+str(args.tar_frac)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)
    file_name += '_seed_'+str(args.seed)
    return net, file_name


def testOnly(args):
    _, file_name = getNetwork(args)
    checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name+'.t7')
    net = checkpoint['net']

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
    return 100.*correct/total


class Trainer:
    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
    args = getArguments()
    #seed_init = int(1E6*random.random())
    #seed_init = 144162
    setSeeds(args.seed)
    
    print('\n[Phase 1] : Data Preparation')
    
    print("| Preparing "+args.dataset+" dataset...")
    sys.stdout.write("| ")
    trainset, testset = getDataset(args)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
    
    def build_net(self, config):
        # self.args.dropout = config['drop_rate']
        # self.args.act_norm = config['q_norm']
        # self.args.tar_frac = config['tar_frac']
        
        print('| Building net type [' + self.args.net_type + ']...')
        self.net, self.file_name = getNetwork(self.args)
        self.net.apply(conv_init)
    
        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
            cudnn.benchmark = True

        self.model_pth='/home/najeeb/dnn_training/bo_compression/exp3/results/'+self.file_name
        # self.model_pth='/mnt/DATA/dnn_training/bo_compression/exp3/results/'+self.file_name
        self.criterion = nn.CrossEntropyLoss()
        
        eligible_params = []
        for name, _ in self.net.named_parameters():
            if 'weight' in name and 'linear' not in name and 'bn' not in name:
                eligible_params.append(name.replace('module.', ''))
        self.pruner = MagnitudePruner(eligible_params)
        print('eligible layers:', eligible_params)
#         self.pruner = MagnitudePruner(exclude_final=True)
        

    # Training
    def train(self, epoch):
        self.net.train()
        train_loss = 0
        correct = 0
        total = 0
        optimizer = optim.SGD(self.net.parameters(), 
                              lr=cf.learning_rate(self.args.lr, epoch), nesterov = True,
                              momentum=0.9, weight_decay=5e-4)
    
        print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(self.args.lr, epoch)))
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
            optimizer.zero_grad()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = self.net(inputs)               # Forward Propagation
            loss = self.criterion(outputs, targets)  # Loss
            loss.backward()  # Backward Propagation
            optimizer.step() # Optimizer update
    
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    
        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, self.num_epochs, batch_idx+1,
                    (len(self.trainset)//self.batch_size)+1, loss.item(), 100.*correct/total))
        sys.stdout.flush()
            
        write(self.model_pth+'_train_loss.csv', train_loss)
        write(self.model_pth+'_train_acc.csv', 100.*correct.item()/total)
            
    
    def test(self, epoch, reporter):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
#         net_clone_pruned = self.get_a_pruned_clone()
        
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                inputs, targets = Variable(inputs), Variable(targets)
                outputs = self.net(inputs)
                loss = self.criterion(outputs, targets)
    
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
    
        # Save checkpoint when best model
        acc = 100.*correct.item()/total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
        write(self.model_pth+'_test_loss.csv', test_loss)
        write(self.model_pth+'_test_acc.csv', acc)
        
#         reporter(timesteps_total=epoch, mean_accuracy=acc)
#         tune.track.log(mean_accuracy=acc)
    
        if acc > self.best_acc:
            print('| Saving Best model...\t\t\tTop1 = %.2f%%' %(acc))
            state = {
                    'net':self.net.module if self.use_cuda else self.net,
                    'acc':acc,
                    'epoch':epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            save_point = './checkpoint/'+self.args.dataset+os.sep
            if not os.path.isdir(save_point):
                os.mkdir(save_point)
            torch.save(state, save_point+self.file_name+'.t7')
            self.best_acc = acc

    def train_itr(self, config, reporter):
        self.build_net(config)
        
        print('\n[Phase 3] : Training model')
        print('| Training Epochs = ' + str(self.num_epochs))
        print('| Initial Learning Rate = ' + str(self.args.lr))
        print('| Optimizer = ' + str(self.optim_type))
        
        elapsed_time = 0
        for epoch in range(self.start_epoch, self.start_epoch+self.num_epochs):
            start_time = time.time()
        
            self.train(epoch)
            self.test(epoch, reporter)
        
            epoch_time = time.time() - start_time
            elapsed_time += epoch_time
            print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))
        self.test_pruned_from_path()
        
        print('* Test results : Acc@1 = %.2f%%' %(self.best_acc))
        
    def load_and_prune_the_model(self, fraction, file_name, prune_units):
        checkpoint = torch.load(file_name+'.t7')
        self.net = checkpoint['net']
        self.pruner.apply(self.net, fraction, prune_units)
        return self.net
        
    
    def test_pruned_from_path(self):
        pruned_fraction = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        file_name = './checkpoint/'+self.args.dataset+os.sep + self.file_name
        
        for prune_units in [False, True]:
            for fraction in pruned_fraction:
                net = self.load_and_prune_the_model(fraction, file_name, prune_units)
                if self.use_cuda:
                    net.cuda()
                    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
                    cudnn.benchmark = True
            
                net.eval()
                test_loss = 0
                correct = 0
                total = 0
                for batch_idx, (inputs, targets) in enumerate(self.testloader):
                    if self.use_cuda:
                        inputs, targets = inputs.cuda(), targets.cuda()
                    with torch.no_grad():
                        inputs, targets = Variable(inputs), Variable(targets)
                        outputs = net(inputs)
            
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += predicted.eq(targets.data).cpu().sum()
                acc = 100.*correct.item()/total
                print('pruning result:', fraction, acc)
                if prune_units:
                    write(self.model_pth+'_test_acc_unit_pruned.csv', acc)
                else:
                    write(self.model_pth+'_test_acc_weight_pruned.csv', acc)
    
            
if __name__ == "__main__":
    t = Trainer()
    # t.train_itr(reporter=None, config={'reg_type':'bp', "drop_rate": 0.5, "q_norm":1.56, 'tar_frac': 0.75})
    t.train_itr(None, None)
    
    
