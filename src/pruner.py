import torch
import math

class MagnitudePruner:
    def __init__(self, exclude_final=True):
        self.exclude_final = exclude_final
            
    def is_param_weight_and_not_bias(self, param_name):
        return 'weight' in param_name and 'bias' not in param_name

    def get_eligible_params(self, net):
        weights = []
        for param_name, param in net.named_parameters():
            if self.is_param_weight_and_not_bias(param_name):
                weights.append(param)
        return weights[:-1] if self.exclude_final else weights
    
    def apply(self, net, fraction):
        weights = self.get_eligible_params(net)
        with torch.no_grad():
            for weight in weights:
                self.prune(weight, fraction)
            
    def prune(self, weight, fraction):
            w_shape = weight.size()
            weight_flattened = weight.view([w_shape[0], -1])
            weight_flattened_abs = torch.abs(weight_flattened)
            sorted_indices = torch.argsort(weight_flattened_abs, dim=1)
            n = int(sorted_indices.size()[1]*fraction)
            threshold_values = weight_flattened_abs.gather(1,sorted_indices)[:,n].view([-1,1])
            mask = weight_flattened_abs.ge(threshold_values)
            weight_flattened.mul_(mask.float())


class MagnitudePrunerLayers:
    @staticmethod
    def is_param_weight_and_not_bias(param_name, eligible_layers):
        result = False
        for l in eligible_layers:
            if l in param_name:
                result = True
                break
        return result and 'bias' not in param_name
    
    @staticmethod
    def get_eligible_params(net, eligible_layers):
        weights = []
        for param_name, param in net.named_parameters():
            if MagnitudePrunerLayers.is_param_weight_and_not_bias(param_name, eligible_layers):
#                 print('added to pruning list ', param_name )
                weights.append(param)
        return weights
    
    @staticmethod
    def apply(net, fraction, eligible_layers):
        weights = MagnitudePrunerLayers.get_eligible_params(net, eligible_layers)
#         print('Before pruning: fraction of non-zeros')
#         print('*'*80)
#         print(MagnitudePrunerLayers.get_non_zero_fraction(net, eligible_layers))
#         print('*'*80)
        
        with torch.no_grad():
            for weight in weights:
                MagnitudePrunerLayers.prune(weight, fraction)
        
#         print('After pruning: fraction of non-zeros')
#         print('*'*80)
#         print(MagnitudePrunerLayers.get_non_zero_fraction(net, eligible_layers))
#         print('*'*80)
                
    @staticmethod
    def prune(weight, fraction):        
        w_shape = weight.size()
        weight_flattened = weight.view([w_shape[0], -1])
        weight_flattened_abs = torch.abs(weight_flattened)
        sorted_indices = torch.argsort(weight_flattened_abs, dim=1)
#         print('sorted_indices.size()[1]', sorted_indices.size()[1])
#         print('sorted_indices.size()[1]*fraction', sorted_indices.size()[1]*fraction)
        n = math.floor(sorted_indices.size()[1]*fraction)
#         print('n', n)
        threshold_values = weight_flattened_abs.gather(1,sorted_indices)[:,n].view([-1,1])
        mask = weight_flattened_abs.ge(threshold_values)
#         print('mask.size()', mask.size())
        weight_flattened.mul_(mask.float())
    
    @staticmethod
    def get_non_zero_fraction(net, eligible_layers):
        non_zero_count = {}
        for param_name, param in net.named_parameters():
            if MagnitudePrunerLayers.is_param_weight_and_not_bias(param_name, eligible_layers):
                non_zero_count[param_name] = param[torch.abs(param)>0].size()[0]#/param.numel()
        return non_zero_count


class MagnitudePrunerLSTM:
    @staticmethod
    def is_param_weight_and_not_bias(param_name):
        return ('fc_gate_x.weight' in param_name or
                'fc_gate_h.weight' in param_name) and 'bias' not in param_name
    
    @staticmethod
    def get_eligible_params(net):
        weights = []
        for param_name, param in net.named_parameters():
            if MagnitudePrunerLSTM.is_param_weight_and_not_bias(param_name):
                weights.append(param)
        return weights
    
    @staticmethod
    def apply(net, fraction):
        weights = MagnitudePrunerLSTM.get_eligible_params(net)
        with torch.no_grad():
            for weight in weights:
                MagnitudePrunerLSTM.prune(weight, fraction)
                
    @staticmethod
    def prune(weight, fraction):
            w_shape = weight.size()
            weight_flattened = weight.view([w_shape[0], -1])
            weight_flattened_abs = torch.abs(weight_flattened)
            sorted_indices = torch.argsort(weight_flattened_abs, dim=1)
            n = int(sorted_indices.size()[1]*fraction)
            threshold_values = weight_flattened_abs.gather(1,sorted_indices)[:,n].view([-1,1])
            mask = weight_flattened_abs.ge(threshold_values)
            weight_flattened.mul_(mask.float())

         
# x = torch.tensor([[[  -3.57467946,   79.54927123,  -96.96720696,  157.74728225, -29.54584318],
#                     [ -55.80235766,   47.82568919,   -1.29289683,  -58.98199797, -69.96299468],
#                     [ 247.83942718,  108.32906629,  -59.86555402,  135.34271468, 19.68149787]],
#                     [[ -78.29762493,  266.53211911,  -43.77736142, -144.96562926, 28.569716  ],
#                      [-115.58819161,  -74.86650242,  -47.24648197, -114.06854625, -88.20043568],
#                      [-130.0732679 , -121.55034213,  202.51414356, -271.76901139, 8.93043837]]])
# 
# prune(x, 0.4)

class UnitPrunerLSTM:
    def __init__(self, exclude_final=True):
        self.exclude_final = exclude_final
            
    @staticmethod
    def is_param_weight_and_not_bias(param_name):
        return ('fc_gate_x.weight' in param_name or
                'fc_gate_h.weight' in param_name) and 'bias' not in param_name

    def get_eligible_params(self, net):
        weights = []
        for param_name, param in net.named_parameters():
            if self.is_param_weight_and_not_bias(param_name):
                weights.append(param)
        return weights[:-1] if self.exclude_final else weights
    
    def apply(self, net, fraction):
        weights = self.get_eligible_params(net)
        with torch.no_grad():
            for weight in weights:
                self.prune(weight, fraction)
            
    def prune(self, weight, fraction):
            w_shape = weight.size()
            weight_flattened = weight.view([w_shape[0], -1])
            unit_l1_norm = torch.norm(weight_flattened, p=1, dim=1)
            sorted_indices = torch.argsort(unit_l1_norm)
            n = int(sorted_indices.size()[0]*fraction)
            threshold = unit_l1_norm[sorted_indices[n]]
            mask = unit_l1_norm.ge(threshold)
            weight_flattened.mul_(mask[:, None].float())


class UnitPruner:
    def __init__(self, exclude_final=True):
        self.exclude_final = exclude_final
            
    def is_param_weight_and_not_bias(self, param_name):
        return 'weight' in param_name and 'bias' not in param_name

    def get_eligible_params(self, net):
        weights = []
        for param_name, param in net.named_parameters():
            if self.is_param_weight_and_not_bias(param_name):
                weights.append(param)
        return weights[:-1] if self.exclude_final else weights
    
    def apply(self, net, fraction):
        weights = self.get_eligible_params(net)
        with torch.no_grad():
            for weight in weights:
                self.prune(weight, fraction)
            
    def prune(self, weight, fraction):
            w_shape = weight.size()
            weight_flattened = weight.view([w_shape[0], -1])
            unit_l1_norm = torch.norm(weight_flattened, p=1, dim=1)
            sorted_indices = torch.argsort(unit_l1_norm)
            n = int(sorted_indices.size()[0]*fraction)
            threshold = unit_l1_norm[sorted_indices[n]]
            mask = unit_l1_norm.ge(threshold)
            weight_flattened.mul_(mask[:, None].float())

def test():
    from sto_reg.tests.net_training.cnn import LeNet5
    x = torch.randn(2,1,28,28)
    pruner = UnitPruner(False)
    net = LeNet5({'name':'LeNet5', 'drop_rate':0.5})
    print(net(x))
    pruner.apply(net, 0.6)
    print(net(x))

if __name__ == '__main__':
    test()
#     from sto_reg.tests.net_training.cnn import LeNet5
#     x = torch.randn(2,1,28,28)
#     pruner = MagnitudePruner(False)
#     net = LeNet5({'name':'LeNet5', 'drop_rate':0.5})
#     print(net(x))
#     pruner.apply(net, 0.6)
#     print(net(x))

    
    
    