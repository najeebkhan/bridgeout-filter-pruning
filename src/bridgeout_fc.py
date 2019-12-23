import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules import Module

class BridgeoutFcLayer(Module):
    r"""Applies the brigdeout transformation to the incoming data: :math:`y = Bx + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        p: dropout probablity
        q: norm of the penalty
        bias: If set to False, the layer will not learn an additive bias.
            Default: True

    Shape:
        - Input: :math:`(N, in\_features)`
        - Output: :math:`(N, out\_features)`

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)

    Examples::

        >>> m = Bridgeout(20, 30)
        >>> input = autograd.Variable(torch.randn(128, 20))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(
            self,
            in_features,
            out_features,
            p=0.5,
            q=2.0,
            bias=True,
            batch_mask=False,
            unit_test_mode=False):
        super(BridgeoutFcLayer, self).__init__()
        self.p=p
        self.q=q / 2.0
        self.in_features = in_features
        self.out_features = out_features
        
        self.unit_test_mode = unit_test_mode
        
        self.rand_gen = torch.Generator()
        if unit_test_mode:            
            self.rand_gen.manual_seed(0)
        
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.use_same_mask = batch_mask
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        if self.unit_test_mode:
            self.rand_gen.manual_seed(0)
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv, generator=self.rand_gen)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv, generator=self.rand_gen)

    def forward(self, input):
        if self.training:
            if self.unit_test_mode:
                self.rand_gen.manual_seed(0)           
            
            bS, inpS = input.size()
            outS = self.weight.size()[1]
            
            input = input.view(bS,1,inpS)
            if not self.use_same_mask:
                w = self.weight.expand(bS, inpS, outS)
            else:
                w = self.weight
            
            
            wq = torch.abs( w ).pow( self.q )
            
            noise = w.data.clone()
            noise.bernoulli_(1 - self.p, generator=self.rand_gen).div_(1 - self.p).sub_(1)
            w = w.add(wq.mul(Variable(noise)))
            
            if self.bias is not None:
                output = input.matmul(w).view(bS,outS).add(self.bias)
            else:
                output = input.matmul(w).view(bS,outS)
        else:
            if self.bias is not None:
                output = input.matmul(self.weight).add(self.bias)
            else:
                output = input.matmul(self.weight)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


if __name__ == '__main__':
    b = BridgeoutFcLayer(2,2, batch_mask=False).double()
    x = Variable(torch.ones(5, 2).double(), requires_grad=True)
    y = b(x)
    y.backward(torch.ones(y.size()).double())
    [print('p, p.grad', n, p.grad) for n, p in b.named_parameters()]
    print(y)
    b.zero_grad()
    y = b(x)
    y.backward(torch.ones(y.size()).double())
    [print('p, p.grad', n, p.grad) for n, p in b.named_parameters()]
    print(y)
    
