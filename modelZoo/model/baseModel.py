import torch 
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):

        self.build_network()
        self.represent = self.build_represent()
        self.loss = self.build_loss()
        self.test_data = self.build_test_data()
    
    def build_represent(self):
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError
    
    def build_CNN(self):
        raise NotImplementedError

    def build_remain(self):
        raise NotImplementedError

    def build_network(self):

        self.cnn = self.build_CNN()
        self.remain = self.build_remain()

    def build_test_data(self):
        raise NotImplementedError

    def forward_network(self, x):
        
        x = self.cnn(x)
        x = self.remain(x)
        return x

    def forward_represent(self, res):
        raise NotImplementedError

    def forward_loss(self, res, target):
        raise NotImplementedError

    def forward(self, x, target):

        loss = 0
        res = "None"
        res = self.forward_network(x)
        
        if self.training:
            loss = self.forward_loss(res, target)
    
        if not self.training:
            res = self.forward_represent(res)

        return res, loss
