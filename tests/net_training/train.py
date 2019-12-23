from sto_reg.tests.net_training.cnn import LeNet5
from sto_reg.tests.net_training.mnist_dataset import get_data_loaders
from tqdm import trange
from sklearn.metrics import accuracy_score
import numpy as np
import torch
import time
import json

class Trainer:
    def __init__(self, arch):
        self.path = 'net_training/trained_data/'
        for k,v in arch.items():
            self.path += str(k) + '_' + str(v) + '_'
        print(self.path)
        self.net = LeNet5(arch)
        self.net.cuda()
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimization = torch.optim.SGD(self.net.parameters(), lr = 0.001, momentum=0.9)
        
        self.training_accuracy = []
        self.validation_accuracy = []
        self.training_loss = []
        self.validation_loss = []
        self.train_loader, self.valid_loader, self.test_loader = get_data_loaders()
    
    def run(self, numEpochs=30):
        start = time.time()
        for epoch in trange(numEpochs):
            self.train(epoch)
            self.validate(epoch)
        total_seconds =  time.time() - start
        self.seconds_per_epoch = total_seconds/numEpochs
        self.save()

    def train(self, epoch):
        
        self.net.train()
        epoch_training_loss = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(self.train_loader):
            inputs, labels = training_batch    
            inputs, labels = torch.autograd.Variable(inputs.cuda()), torch.autograd.Variable(labels.cuda())        
            self.optimization.zero_grad()         
            forward_output = self.net(inputs)
            loss = self.loss_func(forward_output, labels)
            loss.backward()   
            self.optimization.step()     
            epoch_training_loss += loss.item()
            num_batches += 1
        self.training_loss.append(epoch_training_loss/num_batches)

        # calculate training set accuracy
        accuracy = 0.0
        num_batches = 0
        for batch_num, training_batch in enumerate(self.train_loader):            
            num_batches += 1
            inputs, actual_val = training_batch
            predicted_val = self.net(torch.autograd.Variable(inputs.cuda()))
            predicted_val = predicted_val.cpu().data.numpy()    
            predicted_val = np.argmax(predicted_val, axis = 1)  
            accuracy += accuracy_score(actual_val.numpy(), predicted_val)
            
        self.training_accuracy.append(accuracy/num_batches)
#         print("epoch: ", epoch, ", training acc: ", accuracy/num_batches, ", training loss: ",  epoch_training_loss/num_batches)
    
    def validate(self, epoch):
        self.net.eval()
        accuracy = 0.0
        loss = 0.0
        num_batches = 0
        for batch_num, validation_batch in enumerate(self.valid_loader):      
            num_batches += 1
            inputs, actual_val = validation_batch
            predicted_val = self.net(inputs.cuda())
            loss += self.loss_func(predicted_val, actual_val.cuda()).item()
            predicted_val = predicted_val.cpu().data.numpy()
            predicted_val = np.argmax(predicted_val, axis = 1)
            accuracy += accuracy_score(actual_val.cpu().numpy(), predicted_val)
        self.validation_loss.append(loss/num_batches)
        self.validation_accuracy.append(accuracy/num_batches)
#         print("epoch: ", epoch, ", validate acc: ", accuracy/num_batches, ", validate loss: ", loss/num_batches)

    
        
    def save(self):
        meta = "Network:\n"  + str(self.net) +\
        "\n seconds_per_epoch: " + str(self.seconds_per_epoch)
        with open(self.path+'meta', 'w') as f:
            f.write(meta+"\n")
        np.savetxt(self.path+'training_loss', self.training_loss, delimiter=',')
        np.savetxt(self.path+'training_accuracy', self.training_accuracy, delimiter=',')
        np.savetxt(self.path+'validation_accuracy', self.validation_accuracy, delimiter=',')
        np.savetxt(self.path+'validation_loss', self.validation_loss, delimiter=',')
        torch.save(self.net.state_dict(), self.path[:-1]+'.model')
        

if __name__ == "__main__":
    t = Trainer({'name':'LeNet5'})
    t.run(2)
    t.save()
        
        