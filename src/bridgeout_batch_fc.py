import math
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from sto_reg.src.sparseout import SO

class BridgeoutBatchFcLayer(Module):
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
            target_fraction=1.0,
            bias=True,
            unit_test_mode=False):
        super(BridgeoutBatchFcLayer, self).__init__()
       
        self.in_features = in_features
        self.out_features = out_features
        self.unit_test_mode=unit_test_mode
        self.p = p
        self.q = q
        self.tf=target_fraction
        self.dropout = SO(self.p, self.q, self.tf)
        
#         if q:
#             self.dropout = SO(p=p, q=q, target_fraction=target_fraction, unit_test_mode=unit_test_mode)
#         elif p:
#             self.dropout = DO(p=p, target_fraction=target_fraction, unit_test_mode=unit_test_mode)
#         
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        

    def reset_parameters(self):
        self.rand_gen = torch.Generator()
        if self.unit_test_mode:
            self.rand_gen.manual_seed(0)
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv, generator=self.rand_gen)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv, generator=self.rand_gen)

    def forward(self, input_x):
        if self.training:
            bS, _ = input_x.size()
            outS = self.weight.size()[1]
            
            if self.bias is not None:
                output = input_x.matmul(self.dropout(self.weight)).view(bS,outS).add(self.bias)
            else:
                output = input_x.matmul(self.dropout(self.weight)).view(bS,outS)
        else:
            if self.bias is not None:
                output = input_x.matmul(self.weight).add(self.bias)
            else:
                output = input_x.matmul(self.weight)
        
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def testing():
    import time
    from sto_reg.src.sparseout import sparseout
    b = BridgeoutBatchFcLayer(8400,1050, p=0.5, q=1.5, target_fraction=0.75 ).to('cuda')
    stt = time.time()
    for i in range(20):
        x = Variable(torch.rand(20, 8400), requires_grad=True).to('cuda')
        y = b(x)
    dur = time.time()-stt
    print('bo dur', dur*1000)
    
    l = torch.nn.Linear(8400,1050).to('cuda').train()
    stt = time.time()
    for i in range(20):
        x = Variable(torch.rand(20, 8400), requires_grad=True).to('cuda')
        
        l.weight = torch.nn.Parameter(sparseout(l.weight, p=0.5, q=1.8, target_fraction=0.75))
        
        y = l(x)
    print(l.weight.size())
    dur = time.time()-stt
    print('lin dur', dur*1000)
    
    
    print('done')
    

if __name__ == '__main__':
    testing()
#     b = BridgeoutFcLayer(2,2, batch_mask=False, target_fraction=0.75, unit_test_mode=True).double()
#     x = Variable(torch.ones(5, 2).double(), requires_grad=True)
#     y = b(x)
#     y.backward(torch.ones(y.size()).double())
# #     [print('p, p.grad', n, p.grad) for n, p in b.named_parameters()]
#     print(y)
#     b.zero_grad()
#     y = b(x)
#     y.backward(torch.ones(y.size()).double())
#     [print('p, p.grad', n, p.grad) for n, p in b.named_parameters()]
#     print(y)
    
