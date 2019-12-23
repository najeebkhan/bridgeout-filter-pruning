import torch
from torch.autograd import gradcheck
from sto_reg.src.bridgeout_fc import BridgeoutFcLayer
from sto_reg.src.bridgeout_conv import BridgeoutConvLayer
from sto_reg.src.sparseout import SO
from sto_reg.src.dropout import DO
import unittest
 
class TestsBridgeoutFcLayer(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.g = torch.Generator()
        self.bo_layer = BridgeoutFcLayer(
            self.n, self.n, p=0.7, q=1.3, batch_mask=True, unit_test_mode=True).double()
         
    def test_fc_gradients(self):
        self.g.manual_seed(2932)
        input = (torch.randn(
            self.n, self.n, dtype=torch.double,requires_grad=True, generator=self.g))
        test = gradcheck(self.bo_layer, input, eps=1e-6, atol=1e-4)
        assert test == True        
        
class TestsSparseoutLayer(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.g = torch.Generator()
        self.so_layer = SO(p=0.7, q=1.3, target_fraction=0.75, unit_test_mode=True)
        
    def test_so_gradients(self):
        self.g.manual_seed(2872)
        x = torch.randn(self.n, self.n, dtype=torch.double,requires_grad=True, generator=self.g)
        test = gradcheck(self.so_layer, (x,), eps=1e-6, atol=1e-4)
        assert test == True

class TestsDropoutLayer(unittest.TestCase):
    def setUp(self):
        self.n = 20
        self.g = torch.Generator()
        self.do_layer = DO(p=0.7, target_fraction=0.75, unit_test_mode=True)
        
    def test_so_gradients(self):
        self.g.manual_seed(2932)
        x = torch.randn(self.n, self.n, dtype=torch.double,requires_grad=True, generator=self.g)
        test = gradcheck(self.do_layer, (x,), eps=1e-6, atol=1e-4)
        assert test == True

        
        
class TestsBridgeoutConvLayer(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator()
        self.p = 0.7
        self.q = 0.6
        self.n = 2
        self.conv_bo = BridgeoutConvLayer(
            3,1,3, p=self.p, q=self.q, target_fraction=0.75, unit_test_mode=True).double()
         
    def test_gradients(self):
        self.g.manual_seed(2932)
        x = torch.randn(self.n, 3, 10, 10, dtype=torch.double,requires_grad=True, generator=self.g)
        test = gradcheck(self.conv_bo, (x,), eps=1e-6, atol=1e-4)
        assert test == True

if __name__ == '__main__':
    unittest.main()




