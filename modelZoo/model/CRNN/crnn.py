
from ..baseModel import BaseModel

import torch.nn as nn
import torch.nn.functional as F
import torch

from .represent import strLabelConverter

class CRNN(BaseModel):

    def __init__(self, cfg):
        super(CRNN, self).__init__()
        self.cfg = cfg

    def build_represent(self):
        self.represent = strLabelConverter(self.cfg.wordTable.name)

    def build_loss(self):
        self.loss = torch.nn.CTCLoss(reduction='sum')
    
    def build_CNN(self):

        pass

    def build_remain(self):
        
        from .crnnRemain import CRNN
        self.remain = CRNN

    def build_test_data(self):
        pass

    def forward_network(self, x):

        #print('---forward propagation---')
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output

    def forward_represent(self, res):

        batch_size = res.shape[1]
        _, preds = res.max(2)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text_preds = self.strLabelConverter.decode(preds.data, preds_size.data, raw=False)
        return text_preds

    def forward_loss(self, preds, targets):

        batch_size = preds.size()[0]

        target_texts = []
        #if len(targets['texts']) != batch_size:
        for item in targets['texts']:
            target_texts += item
        #else:
        #    target_texts = targets['texts']

        assert len(target_texts) == batch_size

        texts, length = self.strLabelConverter.encode(target_texts)
        preds_size = torch.IntTensor([preds.size(0)]*batch_size)

        recog_loss = self.loss(preds, texts, preds_size, length)/batch_size
        return recog_loss
