from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
from torch.nn.modules import Module

import torch
EPSILON=1E-12
class Sparseout(InplaceFunction):

    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input_x, drop_rate=0.5, so_norm=2.0, target_fraction=1.0, train=False, inplace=False, unit_test_mode=False):
        rand_gen = torch.Generator()
        if unit_test_mode:
            rand_gen.manual_seed(353)
#         EPSILON=1E-32
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(drop_rate))
        if inplace:
            raise NotImplementedError("In place computations haven't been tested yet!")
        ctx.p = drop_rate
        ctx.q = so_norm
        ctx.train = train
        ctx.inplace = inplace
        ctx.input = input_x

        if ctx.inplace:
            ctx.mark_dirty(input_x)
            output = input_x
        else:
            output = input_x.clone()

        if ctx.p > 0 and ctx.train:
            ctx.noise = cls._make_noise(input_x)
            if ctx.p == 1:
                ctx.noise.fill_(0)
                ctx.perturbation = ctx.noise
            else:
                ctx.noise.bernoulli_(1 - ctx.p, generator=rand_gen).div_(1 - ctx.p).sub_(1)
                ctx.perturbation = input_x.abs().add(EPSILON).pow((ctx.q)/2.0).mul(ctx.noise)

            if target_fraction < 1.0:
                input_shape = input_x.size()
                batch_size = input_shape[0]
                input_flattened_abs = torch.abs(input_x.view([batch_size, -1]))
                feature_shape = input_flattened_abs.size()[1]
                n_features_to_drop = int(feature_shape*target_fraction)
                 
                sorted_indices_per_row = torch.argsort(input_flattened_abs, dim=1)
                nth_ranked_feature_value_per_row = input_flattened_abs.gather(1,sorted_indices_per_row)[:,n_features_to_drop].view([-1,1])
  
                ctx.targeting_mask = input_flattened_abs.lt(nth_ranked_feature_value_per_row).view(input_shape).type(input_x.dtype)
                ctx.perturbation.mul_(ctx.targeting_mask)
            output.add_(ctx.perturbation)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            with torch.no_grad():
                tmp = torch.mul( ctx.q/2.0, ctx.input.abs().add(EPSILON).pow((ctx.q/2.0)-1.))
                tmp = tmp.mul(ctx.noise)
                tmp = tmp.mul(torch.sign(ctx.input))
                if hasattr(ctx, 'targeting_mask'):
                    tmp = tmp.mul(ctx.targeting_mask)
                tmp = tmp.add(1.)
            return grad_output.mul(Variable(tmp)), None, None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None, None

def sparseout(input, p, q, target_fraction=1.0, training=False, inplace=False, unit_test_mode=False):
    return Sparseout.apply(input, p, q, target_fraction, training, inplace, unit_test_mode)



class SO(Module):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability *p* using samples from a bernoulli distribution.
    The elements to zero are randomized on every forward call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of *1/(1-p)* during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to True, will do this operation in-place. Default: false

    Shape:
        - Input: `Any`. Input can be of any shape
        - Output: `Same`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = autograd.Variable(torch.randn(20, 16))
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, q=2.0, target_fraction=1.0, inplace=False, unit_test_mode=False):
        super(SO, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.q = q
        self.inplace = inplace
        self.unit_test_mode = unit_test_mode
        self.target_fraction=target_fraction

    def forward(self, input_x):
        return sparseout(input_x, self.p, self.q, self.target_fraction, self.training, self.inplace, self.unit_test_mode)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) \
            + ' q = ' + str(self.q) \
            + ' target_fraction = ' + str(self.target_fraction) \
            + inplace_str + ')'


if __name__ == '__main__':
    x = torch.tensor([[ 0.3439, -0.9], [ -0.439, -1.3], [ 0.9, 0.6]], dtype=torch.double,requires_grad=True)

    x = torch.tensor(
    [[[  -3.57467946,   79.54927123,  -96.96720696,  157.74728225, -29.54584318],
                    [ -55.80235766,   47.82568919,   -1.29289683,  -58.98199797, -69.96299468],
                    [ 247.83942718,  108.32906629,  -59.86555402,  135.34271468, 19.68149787]],
                    [[ -78.29762493,  266.53211911,  -43.77736142, -144.96562926, 28.569716  ],
                     [-115.58819161,  -74.86650242,  -47.24648197, -114.06854625, -88.20043568],
                     [-130.0732679 , -121.55034213,  202.51414356, -271.76901139, 8.93043837]]],
    dtype=torch.double,requires_grad=True)
    x = torch.rand(3,10)*1000

#     x = torch.tensor([[ 10.,   3,   2,  23,   1],
#         [  0,   1,   2,   2,   9],
#         [200,  31,   5,   0,   8],
#         [  9,   1,   0,  10,   2]], dtype=torch.float)

    so = SO(0.5, 1.0, target_fraction=0.34, unit_test_mode=False)
    print(x)

    y = so(x.float())
    
    res = y-x.float()
    print(res.int())
#     print('x-y\n', x.float()-y)
#     print('x\n', x.int())
#     y.backward(torch.ones(y.size()).float(), retain_graph=True)
#     print('y',y)
#     print('-----x.grad', x.grad)
#
#     x.grad.data.zero_()
#     y = so(x)
#     y.backward(torch.tensor([0,1]).double())
#     print('y',y)
#     print('-----x.grad', x.grad)


