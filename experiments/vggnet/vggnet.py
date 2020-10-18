import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
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
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


def cfg(depth):
    depth_lst = [11, 13, 16, 19]
    assert (depth in depth_lst), "Error : VGGnet depth should be either 11, 13, 16, 19"
    cf_dict = {
        '11': [
            64, 'mp',
            128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'],
        '13': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 'mp',
            512, 512, 'mp',
            512, 512, 'mp'
            ],
        '16': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 'mp',
            512, 512, 512, 'mp',
            512, 512, 512, 'mp'
            ],
        '19': [
            64, 64, 'mp',
            128, 128, 'mp',
            256, 256, 256, 256, 'mp',
            512, 512, 512, 512, 'mp',
            512, 512, 512, 512, 'mp'
            ],
    }

    return cf_dict[str(depth)]

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class VGG(nn.Module):
    def __init__(self, depth, num_classes, regularizer):
        super(VGG, self).__init__()
        batchnorm=True
        if regularizer.name == 'dropout':
            self.dropout = DO(p=regularizer.dropout_rate, target_fraction=regularizer.target_fraction)
        elif regularizer.name == 'sparseout':
            self.dropout = SO(p=regularizer.dropout_rate, q=regularizer.q_norm, target_fraction=regularizer.target_fraction)
        else:
            self.dropout = Identity()

        self.features = self._make_layers(cfg(depth), regularizer, batchnorm)
        self.linear = nn.Linear(512, num_classes)

#         print(self)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

    def _make_layers(self, cfg, regularizer, batchnorm):
        layers = []
        in_planes = 3

        for x in cfg:
            if x == 'mp':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batchnorm:
                    layers += [conv3x3(in_planes, x, 1, regularizer), nn.BatchNorm2d(x), nn.ReLU(inplace=True), self.dropout]
                else:
                    layers += [conv3x3(in_planes, x, 1, regularizer), nn.ReLU(inplace=True), self.dropout]
                in_planes = x

        # After cfg convolution
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == "__main__":
    class Params:
     def __init__(self, **kwds):
        self.__dict__.update(kwds)    
    net = VGG(16, 10, Params(name='bo', dropout_rate=0.3, q_norm=1.5, target_fraction=1.0), batchnorm=False)


    # def get_model_footprint_in_mb(model):
    #     return sum(p.element_size()*p.nelement() for p in model.parameters() if p.requires_grad)/(1024*1024)
    # print(get_model_footprint_in_mb(net))

    print(net)

    x = Variable(torch.randn(1,3,32,32), requires_grad=True)
    y = net(x)
    print(y.size())
    x.sum().backward()

