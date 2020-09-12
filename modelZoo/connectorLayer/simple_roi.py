import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import cv2

class SimpleROI(nn.Module):
    
    def __init__(self, shrink_ratio=1, out_height=32, max_ratio=8, channels=3):
        
        self.shrink_ratio = shrink_ratio
        self.out_height = int(out_height*shrink_ratio)
        self.out_width = int(max_ratio*out_height*shrink_ratio)
        self.channels = channels
        self._vis = False
        super(SimpleROI, self).__init__()
        # Spatial transformer localization-network

    def resize_images(self, images):
        
        out_height = self.out_height
        out_width = self.out_width
        channels = self.channels

        res_images = []
        masks = []
        
        for i, img in enumerate(images):

            img = img.unsqueeze(0)
            h, w = np.shape(img)[2:4]
            ratio = out_height / h

            im_arr_resized = F.interpolate(img, (out_height, int(w * ratio)), mode="bilinear")

            #print(im_arr_resized.shape)
            re_h, re_w = im_arr_resized.shape[2:4]

            if re_w >= out_width:
                final_arr = F.interpolate(img,(out_height, out_width), mode="bilinear")

            else:
                final_arr = torch.ones((1, channels, out_height, out_width), dtype=torch.float32).to(img.device)
                final_arr[:,:,:,0:re_w] = im_arr_resized
            
            # 增加mask
            mask = -np.ones((1,re_h,out_width))
            mask[:, :, :re_w] = 1
            
            masks.append(torch.ByteTensor(mask).to(img.device))
            
            if self._vis:
                self.vis(final_arr.squeeze(0).cpu(),i)

            res_images.append(final_arr.squeeze(0))


        return torch.stack(res_images), torch.stack(masks)

    def forward(self, x_batch, boxes_batch):

        shrink_ratio = self.shrink_ratio
        rec_x = []
        batch_sizes = []
      #  print(x_batch.shape)
        for i, boxes in enumerate(boxes_batch):

            x = x_batch[i]
            for box in boxes:
                rec_x += [x[:,int(box[1]*shrink_ratio):int(box[3]*shrink_ratio), 
                              int(box[0]*shrink_ratio):int(box[2]*shrink_ratio)]
                          ]

            batch_sizes.append(len(boxes))

        rec_x, masks = self.resize_images(rec_x)
        return rec_x, batch_sizes

    def vis(self, new_img_torch, id_name):
        img = new_img_torch.numpy().transpose(1,2,0)*255
        cv2.imwrite("/home/wx/tmp_pic/_"+ str(id_name)+'.jpg', img )

if __name__ == "__main__":
    pass



