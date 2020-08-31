
from ..baseModel import BaseModel
from ...convLayer import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import DetectLoss
from .represent import SegDetectorRepresenter

class DB(BaseModel):

    def __init__(self,cfg):

        self.cfg = cfg
        super(DB, self).__init__()
    
    def build_represent(self):

        represent = SegDetectorRepresenter()
        return represent

    def build_loss(self):
        
        detect_loss = DetectLoss()
        return detect_loss
    
    def build_CNN(self):

        return FPN(resnet50())

    def build_remain(self):

        from .DBremain import DBSegDetector
        return DBSegDetector()

    def build_test_data(self):
        return torch.FloatTensor((10,3,1024,2048))

    def forward_represent(self, res):

        segmentation, boxes_batch, scores_batch = self.represent.represent(res['binary'], (1<<10, 1<<11))

        res.update(segmentation=segmentation, boxes_batch=boxes_batch,
                   scores_batch=scores_batch)
        
        return res

    def forward_loss(self, res, target):

        loss = self.loss(res, target)
        return loss