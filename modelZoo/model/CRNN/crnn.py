
from ..baseModel import BaseModel
from ...convLayer import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from .crnnRemain import CRNN_remain


from .represent import strLabelConverter

class CRNN(BaseModel):

    def __init__(self, cfg):
        self.cfg = cfg
        super(CRNN, self).__init__()

    def build_represent(self):
        return strLabelConverter(self.cfg.alphabet)

    def build_loss(self):
        return torch.nn.CTCLoss(reduction='sum')
    
    def build_CNN(self):
        
        return FPN(resnet50())

    def build_remain(self):
        
        return CRNN_remain(imgH=8, nc=256, nclass = len(self.represent))

    def build_test_data(self):
        
        return torch.FloatTensor((10,3,32,320))

    def forward_represent(self, res):
        
        batch_size = res.shape[1]
        _, preds = res.max(2)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        text_preds = self.represent.decode(preds.data, preds_size.data, raw=False)
        return text_preds

    def forward_loss(self, preds, targets):

        batch_size = preds.shape[1]

        target_texts = []
        #if len(targets['texts']) != batch_size:
        for item in targets['texts']:
            target_texts += item['texts']
        #else:
        #    target_texts = targets['texts']
        assert len(target_texts) == batch_size

        texts, length = self.represent.encode(target_texts)
        preds_size = torch.IntTensor([preds.size(0)]*batch_size)
        recog_loss = self.loss(preds, texts, preds_size, length)/batch_size
        return recog_loss
