import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import sys
from sto_reg.src.sparseout import SO
from sto_reg.src.dropout import DO
from sto_reg.src.bridgeout_conv import BridgeoutConvLayer
import numpy as np
def conv3x3(in_planes, out_planes, stride, regularizer):
    if regularizer.name in ['dropout', 'sparseout', 'bp']:
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,  padding=1, bias=True)
    else:
        return BridgeoutConvLayer(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=3,
            p=regularizer.dropout_rate,
            q=regularizer.q_norm,
            target_fraction=regularizer.target_fraction,
            stride=stride,
            padding=1)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)

def cfg(depth):
    depth_lst = [18, 34]#, 50, 101, 152]
    assert (depth in depth_lst), "Error : Resnet depth should be either 18, 34" #, 50, 101, 152"
    cf_dict = {
        '18': (BasicBlock, [2, 2, 2, 2]),
        '34': (BasicBlock, [3, 4, 6, 3]),
        '50': (Bottleneck, [3, 4, 6, 3]),
        '101':(Bottleneck, [3, 4, 23, 3]),
        '152':(Bottleneck, [3, 8, 36, 3]),
    }

    return cf_dict[str(depth)]
def identity_fcn(x): return x

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, regularizer, batchnorm):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride, regularizer)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            
        self.conv2 = conv3x3(planes, planes, stride=1, regularizer=regularizer)

        if regularizer.name == 'dropout':
            self.dropout = DO(p=regularizer.dropout_rate, target_fraction=regularizer.target_fraction)
        elif regularizer.name == 'sparseout':
            self.dropout = SO(p=regularizer.dropout_rate, q=regularizer.q_norm, target_fraction=regularizer.target_fraction)
        else:
            self.dropout = Identity()
                              
                            
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if batchnorm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                    Identity()
                )

    def forward(self, x):
        out = self.dropout(
                           F.relu(
                             self.bn1(
                                      self.conv1(x)
                                      )
                             )
                           )
        
        out = self.dropout(
                           self.bn2(
                                    self.conv2(out)
                                    )
                           )
        
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, reg_type, batchnorm):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=True)
        
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
            self.bn2 = nn.BatchNorm2d(planes)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        else:
            self.bn1 = Identity()
            self.bn2 = Identity()
            self.bn3 = Identity()
            

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if batchnorm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                    nn.BatchNorm2d(self.expansion * planes)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=True),
                    Identity()
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, depth, num_classes, reg_type, batchnorm=True):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.reg_type = reg_type

        block, num_blocks = cfg(depth)

        self.conv1 = conv3x3(3, 16, stride=1, regularizer=reg_type)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(16)
        else:
            self.bn1 = Identity()
            
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, batchnorm=batchnorm)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, batchnorm=batchnorm)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, batchnorm=batchnorm)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, batchnorm):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.reg_type, batchnorm=batchnorm))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    class Params:
     def __init__(self, **kwds):
        self.__dict__.update(kwds)    
    net = ResNet(34, 10, Params(name='bo', dropout_rate=0.3, q_norm=1.5, target_fraction=1.0), batchnorm=False)
    print(net)
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())
