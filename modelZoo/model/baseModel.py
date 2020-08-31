import torch 
import torch.nn as nn

class BaseModel(nn.Module):

    def __init__(self):

        super(BaseModel, self).__init__()
        
        self.represent = self.build_represent()
        self.build_network()
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

    def forward(self, x, target= None):

        loss = 0
        res = "None"
        res = self.forward_network(x)
        
        if self.training:
            loss = self.forward_loss(res, target)
            return res, loss

    
        if not self.training:
            res = self.forward_represent(res)
            return res

