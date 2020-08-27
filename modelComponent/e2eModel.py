import torch
import torch.nn as nn

class E2EModel(nn.Module):

    shrink_ratio = 0.25
    keys = 'VAT'
    end2end = True
    _inference_using_bounding_box = False
    run_s1 = True
    run_s2 = True

    def __init__(self, taskConfig, shareConv, modelS1, S1representer, modelS2, S2representer, connector, expandS2=None):
        
        super(E2EModel, self).__init__()

        self.share_conv = shareConv
        self.S1model_branch = modelS1
        self.S2model_branch = modelS2

        self.S1representer = S1representer()
        self.S2representer = S2representer(self.keys)   

        self.connector = connector
        nclass = len(self.S2representer) 
        
        super(E2EModel, self).__init__()

    def forward():
        pass


    # 只运行detect
    def only_run_detect(self):

        self.run_detect = True
        self.run_rec = False
    
    # 只运行rec
    def only_run_rec(self):
        
        self.run_detect = False
        self.run_rec = True
    
    def freeze_conv_layer(self):

        for para in self.share_conv.parameters():
            para.requires_grad = False

    def detect_loss(self, detect_res, targets):
        
        detect_loss = DetectLoss()
        loss = detect_loss(detect_res, targets)
        return loss

    def recog_loss(self, preds, targets, batch_size):
        
        ctc_loss = torch.nn.CTCLoss(reduction='sum')

        target_texts = []
        #if len(targets['texts']) != batch_size:
        for item in targets['texts']:
            target_texts += item
        #else:
        #    target_texts = targets['texts']

        assert len(target_texts) == batch_size

        texts, length = self.strLabelConverter.encode(target_texts)
        preds_size = torch.IntTensor([preds.size(0)]*batch_size)

        recog_loss = ctc_loss(preds, texts, preds_size, length)/batch_size
        return recog_loss

    def forward_detect(self, x, detect_target=None):
        
        detect_loss = 0
        detect_res = "None"

        if self.run_detect:
            batch_size = x.size()[0]
            conv_maps = self.share_conv(x)
  
            detect_res = self.detect_branch(conv_maps)
            if self.training:

                detect_loss = self.detect_loss(detect_res, detect_target)
        
        if self.end2end:
            detect_res['conv_maps'] = conv_maps

        if not self.training:
            detect_res = self.represent_detect(detect_res)

        return detect_res, detect_loss

    def represent_detect(self, detect_res):
        
        segmentation, boxes_batch, scores_batch = self.box_represent.represent(detect_res['binary'], (1<<10, 1<<11))
        
        detect_res.update(segmentation=segmentation, boxes_batch=boxes_batch,
                   scores_batch=scores_batch)

        return detect_res

    # 送出的batches 要恢复原状
    def _unsqueeze_batches(self, list_batches, batch_sizes):
        
        res_batches = []
        start = 0

        for size in batch_sizes:
            res_b = list_batches[start:start+size]
            res_batches.append(res_b)
            start += size

        assert start == len(list_batches)
        return res_batches

    def forward_rec(self, x_batch, rec_targets = None):

        # rec_x 的shape [ batch_size * mini_batch_size, 256, h,w]
        rec_loss = 0
        recog_res = "None"
        
        if self.run_rec:
            batch_size = x_batch.size()[0]
            if not self.end2end:
                #x_batch = self.aug(x_batch)
                x_batch = self.share_conv(x_batch)

            recog_res = self.recog_branch(x_batch)

        if self.training:
            rec_loss = self.recog_loss(recog_res, rec_targets, batch_size)
        else:
            recog_res = self.repreasent_txts(recog_res)

        return recog_res, rec_loss

    def repreasent_txts(self, recog_res):
        
        batch_size = recog_res.shape[1]
        _, preds = recog_res.max(2)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        preds = preds.transpose(1, 0).contiguous().view(-1)
        text_preds = self.strLabelConverter.decode(preds.data, preds_size.data, raw=False)
        return text_preds

    def forward(self, x_batch, detect_target=None, boxes_batch=None, rec_target=None):
        
        detect_res, detect_loss = self.forward_detect(x_batch, detect_target)

        if not self.training:
            if not self._inference_using_bounding_box:
                boxes_batch = detect_res['boxes_batch']

        if self.end2end:
            x_batch = detect_res['conv_maps']
        
        # 把 conv_layer 或 原图 进行roi 裁剪 得到一个batch 送给识别
        rec_x, batch_sizes = self.roi(x_batch, boxes_batch)

        recog_res, rec_loss = self.forward_rec(rec_x, rec_target)
        
        if not self.training:
            recog_res = self._unsqueeze_batches(recog_res, batch_sizes)

        return detect_res, recog_res, detect_loss, rec_loss

    def forward_mutiTask(self, x, detect_target=None, rec_images=None, rec_targets=None):
        
        detect_res, detect_loss = self.forward_detect(x, detect_target)
        recog_res, rec_loss = self.forward_rec(rec_images, rec_targets)
        
        return detect_res, recog_res, detect_loss, recog_loss

###################################################
