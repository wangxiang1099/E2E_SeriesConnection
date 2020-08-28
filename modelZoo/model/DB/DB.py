
from ..baseModel import BaseModel

import torch.nn as nn
import torch.nn.functional as F

class DB(nn.Module):

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
        self.cnn = FPN(resnet18(), in_channels=[64,128,256,512])

    def build_remain(self):
        from .DBremain import DBSegDetector
        self.remain = DBSegDetector()

    def build_test_data(self):
        raise NotImplementedError

    def forward_represent(self, res):
        raise NotImplementedError

    def forward_loss(self, res, target):
        raise NotImplementedError