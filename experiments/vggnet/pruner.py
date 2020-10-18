import torch

class MagnitudePruner:
    def __init__(self, eligible_layers):
        self.eligible_layers = eligible_layers
    
    def get_eligible_params(self, net):
        weights = []
        for param_name, param in net.named_parameters():
            if param_name in self.eligible_layers:
                weights.append(param)
        return weights
    
    def apply(self, net, fraction, prune_units=False):
        weights = self.get_eligible_params(net)
        with torch.no_grad():
            for weight in weights:
                if prune_units:
                    self.prune_units(weight, fraction)
                else:
                    self.prune_weights(weight, fraction)
            self.print_non_zero_fraction(net)

    def prune_weights(self, weight, fraction):
        output_shape = weight.size()[0]
        weight_flattened = weight.view([output_shape, -1])
        feature_shape = weight_flattened.size()[1]
        n_weights_to_drop = int(feature_shape*fraction)
        weight_flattened_abs = torch.abs(weight_flattened)
        sorted_indices_per_row = torch.argsort(weight_flattened_abs, dim=1)                        
        gathered_result = weight_flattened_abs.gather(1,sorted_indices_per_row)
        nth_ranked_weight_value_per_row = gathered_result[:,n_weights_to_drop].view([-1,1])            
        mask = weight_flattened_abs.ge(nth_ranked_weight_value_per_row)            
        weight_flattened.mul_(mask.float())
                       
    def prune_units(self, weight, fraction):
        output_shape = weight.size()[0]
        n_units_to_drop = int(output_shape*fraction)
        weight_flattened = weight.view([output_shape, -1])
        unit_l2_norm = torch.norm(weight_flattened, p=2, dim=1)
        sorted_indices = torch.argsort(unit_l2_norm)
        nth_ranked_unit_l2_value = unit_l2_norm[sorted_indices[n_units_to_drop]]
        mask = unit_l2_norm.ge(nth_ranked_unit_l2_value)
        weight_flattened.mul_(mask[:, None].float())
            
    def print_non_zero_fraction(self, net):
        non_zero_count = {}
        for param_name, param in net.named_parameters():
            if param_name in self.eligible_layers:
                non_zero_count[param_name] = param[torch.abs(param)>0].size()[0]#/param.numel()
                print('layer:', param_name, '\tnon-zero weights:', non_zero_count[param_name], '\tnon-zero fraction:', param[torch.abs(param)>0].size()[0]/param.numel())
#         return non_zero_count

            

def test():
    from sto_reg.tests.net_training.cnn import LeNet5
    x = torch.randn(2,1,28,28)
    pruner = MagnitudePruner(['conv1.weight'])
    net = LeNet5({'name':'LeNet5', 'drop_rate':0.5})
    w = torch.rand(3,8)-0.5
    w[1,:]=w[1,:]+5
    print(w)
    net.fc1.weight = torch.nn.Parameter(
        w)
#         torch.tensor([
#         [ 1.,   3,   -2,  23,   1],
#         [  0,   2.5,   -2.5,   2,   9],
#         [10,  31,   5.2,   0,   5],
#         [200,  1,   5.1,   3,   8.9],
#         [20,  3.5,   5.9,   0,   8],
#         [  9,   1.5,   0,  10,   2]]))
# #     print(net(x))
#     print('net.fc1.weight', net.fc1.weight)
    pruner.apply(net, 0.5, False)
    print(net.conv1.weight)
    print('*'*80)
    
    
#     print(net(x))

if __name__ == '__main__':
    test()
#     from sto_reg.tests.net_training.cnn import LeNet5
#     x = torch.randn(2,1,28,28)
#     pruner = MagnitudePruner(False)
#     net = LeNet5({'name':'LeNet5', 'drop_rate':0.5})
#     print(net(x))
#     pruner.apply(net, 0.6)
#     print(net(x))

    
    
    