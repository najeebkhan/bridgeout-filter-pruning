from torch.autograd.function import InplaceFunction
from torch.autograd import Variable
from torch.nn.modules import Module
import torch
class Dropout(InplaceFunction):
    
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input_x, drop_rate=0.5, target_fraction=1.0, train=False, inplace=False, unit_test_mode=False):
        rand_gen = torch.Generator()
        if unit_test_mode:
            rand_gen.manual_seed(353)
        if drop_rate < 0 or drop_rate > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(drop_rate))
        if inplace:
            raise NotImplementedError("In place computations haven't been tested yet!")
        ctx.p = drop_rate
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
            else:
                ctx.noise.bernoulli_(1 - ctx.p, generator=rand_gen).div_(1 - ctx.p)

            is_filter_map = False
            if input_x.dim() > 3:
                is_filter_map = True

            if target_fraction < 1.0:
                if is_filter_map:
                    input_shape = input_x.size()
                    batch_size = input_shape[0]
                    num_filters = input_shape[1]

                    input_flattened_abs = torch.norm(input_x.view([batch_size, num_filters, -1]), 2, dim=2)
                    feature_shape = input_flattened_abs.size()[1]

                    n_features_to_drop = int(feature_shape*target_fraction)
                    sorted_indices_per_row = torch.argsort(input_flattened_abs, dim=1)
                    nth_ranked_feature_value_per_row = input_flattened_abs.gather(1,sorted_indices_per_row)[:,n_features_to_drop].view([-1,1])
                    targeting_mask = input_flattened_abs.lt(nth_ranked_feature_value_per_row)[:,:,None,None] # .view(input_shape)
                    # print('targeting_mask',targeting_mask)
                    ctx.noise = ctx.noise.where(targeting_mask, torch.tensor([1.0]).type(input_x.dtype).to(input_x.device))
                else:
                    input_shape = input_x.size()
                    batch_size = input_shape[0]
                    input_flattened_abs = torch.abs(input_x.view([batch_size, -1]))
                    feature_shape = input_flattened_abs.size()[1]

                    n_features_to_drop = int(feature_shape*target_fraction)
                    sorted_indices_per_column = torch.argsort(input_flattened_abs, dim=1)
                    nth_ranked_feature_value_per_column = input_flattened_abs.gather(1,sorted_indices_per_column)[:,n_features_to_drop].view([-1,1])
                    targeting_mask = input_flattened_abs.lt(nth_ranked_feature_value_per_column).view(input_shape)
                    print(targeting_mask)
                    ctx.noise = ctx.noise.where(targeting_mask, torch.tensor([1.0]).type(input_x.dtype).to(input_x.device))

            output.mul_(ctx.noise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.train:
            return grad_output.mul(Variable(ctx.noise)), None, None, None, None, None
        else:
            return grad_output, None, None, None, None, None

def dropout(input, p, target_fraction=1.0, training=False, inplace=False, unit_test_mode=False):
    return Dropout.apply(input, p, target_fraction, training, inplace, unit_test_mode)



class DO(Module):
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
        #
        # >>> m = nn.Dropout(p=0.2)
        # >>> input = autograd.Variable(torch.randn(20, 16))
        # >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def __init__(self, p=0.5, target_fraction=1.0, inplace=False, unit_test_mode=False):
        super(DO, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace
        self.unit_test_mode = unit_test_mode
        self.target_fraction=target_fraction

    def forward(self, input_x):
        return dropout(input_x, self.p, self.target_fraction, self.training, self.inplace, self.unit_test_mode)

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' \
            + 'p = ' + str(self.p) \
            + ' target_fraction = ' + str(self.target_fraction) \
            + inplace_str + ')'


if __name__ == '__main__':
    # x = torch.tensor([[ 0.3439, -0.9], [ -0.439, -1.3], [ 0.9, 0.6]], dtype=torch.double,requires_grad=True)
    #
    # x = torch.tensor(
    # [[[  -3.57467946,   79.54927123,  -96.96720696,  157.74728225, -29.54584318],
    #                 [ -55.80235766,   47.82568919,   -1.29289683,  -58.98199797, -69.96299468],
    #                 [ 247.83942718,  108.32906629,  -59.86555402,  135.34271468, 19.68149787]],
    #                 [[ -78.29762493,  266.53211911,  -43.77736142, -144.96562926, 28.569716  ],
    #                  [-115.58819161,  -74.86650242,  -47.24648197, -114.06854625, -88.20043568],
    #                  [-130.0732679 , -121.55034213,  202.51414356, -271.76901139, 8.93043837]]],
    # dtype=torch.double,requires_grad=True)
    #
    # x = torch.tensor([[ 10.,   3,   2,  23,   1],
    #     [  0,   1,   2,   2,   9],
    #     [200,  31,   5,   0,   8],
    #     [  9,   1,   0,  10,   2]], dtype=torch.float)
    # x = torch.rand(3,10)
    #
    # so = DO(0.5, target_fraction=0.65, unit_test_mode=False)
    # print(x)
    #
    # y = so(x.float())
    # print(y)

    x = torch.rand(4,6, 2, 2)*1000
    do = DO(1.0, target_fraction=0.67, unit_test_mode=False)
    # print(x)
    y = do(x.float())

    print(y.int())
    print('In the above out of 6, 4 activation maps are targeted.')
    print('*'*80)

    x = torch.rand(4, 3, 3)*1000
    y = do(x.float())

    print(y.int())
    print('In the above 2/3 of each 3x3 kernel is targeted')

    
    
    
#     so = DO(0.25, target_fraction=0.25, unit_test_mode=False)
#     do = torch.nn.Dropout(0.25)
#     
#     y = so(x.float())
#     y.backward(torch.ones(y.size()).float(), retain_graph=True)
#     
#     print('x-y\n', y-x.float())
#     print('x\n', x.int())
    
    
#     y = do(x.float())
#     print('x-y\n', y-x.float())
#     y.backward(torch.tensor([1,0]).double(), retain_graph=True)
#     print('y',y)
#     print('-----x.grad', x.grad)
#     
#     x.grad.data.zero_()
#     y = so(x)
#     y.backward(torch.tensor([0,1]).double())
#     print('y',y)
#     print('-----x.grad', x.grad)
    

