from sto_reg.tests.net_training.cnn import LeNet5
from sto_reg.src.pruner import MagnitudePruner, UnitPruner
import torch
import unittest

class TestsPruning(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(2,1,28,28).cuda()
        self.net = LeNet5({'name':'LeNet5', 'drop_rate':0.5}).cuda()
         
    def test_magnitude_pruning(self):
        FRACTION_PRUNED = 0.64
        mag_pruner = MagnitudePruner(exclude_final=False)
        mag_pruner.apply(self.net, FRACTION_PRUNED)
         
        non_zero_fraction = self.get_non_zero_fraction()
         
        for name, fraction in non_zero_fraction.items():
            if 'weight' in name:
                print(name)
                self.assertAlmostEqual(fraction, 1-FRACTION_PRUNED, places=2)
#                 
    def test_unit_pruning(self):
        FRACTION_PRUNED = 0.35
        unit_pruner = UnitPruner(exclude_final=False)
        unit_pruner.apply(self.net, FRACTION_PRUNED)
        
        non_zero_fraction = self.get_non_zero_fraction()
        
        for name, fraction in non_zero_fraction.items():
            if 'weight' in name:
                print(name, 'non zeor fraction', fraction)
#                 self.assertAlmostEqual(fraction, 1-FRACTION_PRUNED, places=3)

    def get_non_zero_fraction(self):
        non_zero_count = {}
        for param_name, param in self.net.named_parameters():
            non_zero_count[param_name] = param[torch.abs(param)>0].size()[0]/param.numel()
        return non_zero_count
    
    
if __name__ == '__main__':
    unittest.main()