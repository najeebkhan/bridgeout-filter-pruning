import torch
from sto_reg.src.bridgeout_conv import BridgeoutConvLayer
from sto_reg.src.bridgeout_fc import BridgeoutFcLayer
from sto_reg.src.sparseout import SO
import os

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class LeNet5(torch.nn.Module):

    def __init__(self, arch):
        super(LeNet5, self).__init__()
        assert arch['name'] in ['LeNet5_SO_Conv', 'LeNet5_SO_Fc', 'LeNet5_SO_All',
                                'LeNet5_BO_Conv', 'LeNet5_BO_Fc', 'LeNet5_BO_All',
                                'LeNet5_DO_Conv', 'LeNet5_DO_Fc', 'LeNet5_DO_All', 'LeNet5'], 'Unkown architecture name'
        # TODO
        self.path = 'net_training/trained_data/'
        
        for k,v in arch.items():
            self.path += str(k) + '_' + str(v) + '_'
        if os.path.exists(self.path + 'training_hmeasure'):
            os.remove(self.path + 'training_hmeasure')
        if os.path.exists(self.path + 'validation_hmeasure'):
            os.remove(self.path + 'validation_hmeasure')
        self.dropout_conv1 = Identity()
        self.dropout_conv2 = Identity()
        self.dropout_fc1 = Identity()
        self.dropout_fc2 = Identity()                        
        
        # In LeNet-5, 32x32 images are given as input. Hence padding of 2 is done below            
        if 'BO_All' in arch['name'] or 'BO_Conv' in arch['name']:
            self.conv1 = BridgeoutConvLayer(
                in_channels=1,
                out_channels=6,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True,
                p=arch['drop_rate'],
                q=arch['bo_norm'])
            self.conv2 = BridgeoutConvLayer(
                in_channels=6,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0,
                bias=True,
                p=arch['drop_rate'],
                q=arch['bo_norm'])
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
            self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
       
        
        if 'SO_Conv' in arch['name']:
            self.dropout_conv1 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            self.dropout_conv2 = SO(p=arch['drop_rate'], q=arch['bo_norm'])            
        if 'SO_Fc' in arch['name']:
            self.dropout_fc1 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            self.dropout_fc2 = SO(p=arch['drop_rate'], q=arch['bo_norm'])        
        if 'SO_All' in arch['name']:
            self.dropout_conv1 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            self.dropout_conv2 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            self.dropout_fc1 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            self.dropout_fc2 = SO(p=arch['drop_rate'], q=arch['bo_norm'])
            
        if 'DO_Conv' in arch['name']:
            self.dropout_conv1 = torch.nn.Dropout(p=arch['drop_rate'])
            self.dropout_conv2 = torch.nn.Dropout(p=arch['drop_rate'])            
        if 'DO_Fc' in arch['name']:
            self.dropout_fc1 = torch.nn.Dropout(p=arch['drop_rate'])
            self.dropout_fc2 = torch.nn.Dropout(p=arch['drop_rate'])        
        if 'DO_All' in arch['name']:
            self.dropout_conv1 = torch.nn.Dropout(p=arch['drop_rate'])
            self.dropout_conv2 = torch.nn.Dropout(p=arch['drop_rate'])
            self.dropout_fc1 = torch.nn.Dropout(p=arch['drop_rate'])
            self.dropout_fc2 = torch.nn.Dropout(p=arch['drop_rate'])
       
            
        self.max_pool_1 = torch.nn.MaxPool2d(kernel_size=2)
        self.max_pool_2 = torch.nn.MaxPool2d(kernel_size=2)
        
        
        if 'BO_All' in arch['name'] or 'BO_Fc' in arch['name']:
            self.fc1 = BridgeoutFcLayer(16 * 5 * 5, 120, p=arch['drop_rate'], q=arch['bo_norm'], batch_mask=True)
            self.fc2 = BridgeoutFcLayer(120, 84, p=arch['drop_rate'], q=arch['bo_norm'], batch_mask=True)
        else:
            self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
            self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)
        
    def forward(self, x):
        hmeasure = []
        x = torch.nn.functional.relu(self.conv1(x))
        x = self.dropout_conv1(x)
        if not self.training:
            hmeasure.append(self.hoyer_measure(x))        
        x = self.max_pool_1(x)
        
        x = torch.nn.functional.relu(self.conv2(x))
        x = self.dropout_conv2(x)
        if not self.training:
            hmeasure.append(self.hoyer_measure(x))
        x = self.max_pool_2(x)

        x = x.view(-1, 16 * 5 * 5)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        if not self.training:
            hmeasure.append(self.hoyer_measure(x))
        
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout_fc2(x)
        if not self.training:
            hmeasure.append(self.hoyer_measure(x))
                
        x = self.fc3(x)
        if not self.training:
            hmeasure.append(self.hoyer_measure(x))
            
        if not self.training:
            self.log_sparsity(hmeasure)
        return x
    
    def log_sparsity(self, hmeasures):
        fname = self.path + 'training_hmeasure' if self.training else self.path + 'validation_hmeasure'
        curpath = os.path.abspath(os.curdir)
        with open(fname, 'a+') as f:
            f.write(", ".join([str(i) for i in hmeasures])+"\n")
        
    
    @staticmethod
    def hoyer_measure(x):
        r"""
            Hoyer's measure of sparsity
        """
        with torch.no_grad():
            tmp1 = torch.norm(x, 1)/(torch.norm(x, 2)+1E-12)
            nel = torch.FloatTensor([x.numel()])
            tmp2 = (torch.sqrt(nel) - tmp1)/(torch.sqrt(nel) - 1)
            return tmp2.item()


if __name__ == '__main__':
    l = LeNet5({'name':'LeNet5_BO_Conv', 'drop_rate':0.5, 'bo_norm':1.8})
    c = l(torch.randn(1, 1, 28, 28))
    print(c)
