import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import numpy as np
from sto_reg.src.sparseout import SO
from sto_reg.src.dropout import DO
from sto_reg.src.bridgeout_conv import BridgeoutConvLayer


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
        
def identity_fcn(x): return x

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, regularizer, stride=1):
        super(wide_basic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)

        self.conv1 = conv3x3(in_planes, planes, stride=1, regularizer=regularizer)
        
        if regularizer.name == 'dropout':
            self.dropout = DO(p=regularizer.dropout_rate, target_fraction=regularizer.target_fraction)
        elif regularizer.name == 'sparseout':
            self.dropout = SO(p=regularizer.dropout_rate, q=regularizer.q_norm, target_fraction=regularizer.target_fraction)
        else:
            self.dropout = identity_fcn

        self.bn2 = nn.BatchNorm2d(planes)

        self.conv2 = conv3x3(planes, planes, stride=stride, regularizer=regularizer)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(
                           self.conv1(F.relu(self.bn1(x)))
                           )
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes, regularizer):
        super(Wide_ResNet, self).__init__()
        self.in_planes = 16

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('| Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0], stride=1, regularizer=regularizer)
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, regularizer, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, regularizer, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, regularizer, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, regularizer, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, regularizer, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

if __name__ == '__main__':
    class Params:
     def __init__(self, **kwds):
        self.__dict__.update(kwds)
    net=Wide_ResNet(28, 10, 10, regularizer=Params(name='sparseout', dropout_rate=0.3, q_norm=1.5))
    print(net)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
