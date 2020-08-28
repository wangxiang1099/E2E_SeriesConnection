import torch
import torch.nn as nn

class E2EModel(nn.Module):

    shrink_ratio = 0.25
    end2end = True
    _inference_using_bounding_box = False
    run_s1 = True
    run_s2 = True

    def __init__(self, taskConfig, shareConv, modelS1, modelS2, connector, expandS2=None):
        
        super(E2EModel, self).__init__()

        self.share_conv = shareConv
        self.S1model_branch = modelS1
        self.S2model_branch = modelS2

        self.connector = connector
        
        super(E2EModel, self).__init__()

    def forward(self, x_batch, s1_target=None, s2_target=None):
        
        conv_maps = self.share_conv(x_batch)

        if self.training:
            s1_res, s1_loss = self.S1model_branch(conv_maps, s1_target)
        else:  
            s1_res = self.S1model_branch(conv_maps)

        if not self.training:
            if not self._inference_using_bounding_box:
                boxes_batch = s1_res['boxes_batch']
            else:
                boxes_batch = s1_target['boxes_batch']
        
        # 把 conv_layer 或 原图 进行roi 裁剪 得到一个batch 送给识别
        conv_maps_transforms, batch_sizes = self.connector(conv_maps, boxes_batch)

        if self.training:
            s2_res, s2_loss = self.S2model_branch(conv_maps_transforms, s2_target)
        else:  
            s2_res = self.S2model_branch(conv_maps_transforms)
            s2_res = self._unsqueeze_batches(s2_res, batch_sizes)

        return s1_res, s2_res, s1_loss, s2_loss

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

###################################################
