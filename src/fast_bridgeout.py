import torch
from torch import nn
from torch.distributions.normal import Normal
import math

class FastBridgeout(nn.Module):
    def __init__(self, input_features, output_features, dropout=0.5, bo_norm=2.0, activ='ReLU', unit_test_mode=False):
        assert activ=='ReLU', "Only ReLU activations are supported at the moment."
        if unit_test_mode:
            seed = 979
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        super(FastBridgeout, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.keep_prob = 1. - dropout
        self.bo_norm = bo_norm
        self.activ = activ
        self.unit_test_mode = unit_test_mode

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.bias = nn.Parameter(torch.Tensor(output_features))       
        self.reset_parameters()
            
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)


    def forward(self, input):
        if self.unit_test_mode:
            seed = 979
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        mean_out = input.mm(self.weight.t())
        mean_out += self.bias.unsqueeze(0).expand_as(mean_out)

        tmp = input.pow(2).mul((1 - self.keep_prob)/self.keep_prob)
        variance_out = tmp.mm(torch.abs(self.weight.t())**self.bo_norm)
        
        r = mean_out.div(variance_out)
        dist = Normal(torch.zeros_like(mean_out), torch.ones_like(mean_out))
        
        return mean_out.mul(dist.cdf(r)).add(
            torch.sqrt(variance_out).mul(torch.exp(dist.log_prob(r)))
            )


    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
        
if __name__ == '__main__':
    from torch.autograd import gradcheck
    fd_layer = FastBridgeout(4,3, unit_test_mode=True).double()
    g = torch.Generator()
    g.manual_seed(2345)
    x = torch.randn(5,4, dtype=torch.double,requires_grad=True, generator=g)
    test = gradcheck(fd_layer, (x,), eps=1e-3, atol=1e-2)
    print(test)
