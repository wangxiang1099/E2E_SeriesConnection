import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import numpy as np
import cv2

class QuadROI(nn.Module):
    def __init__(self, shrink_ratio=1, out_height=32, max_ratio=8, channels=3):
        
        self.shrink_ratio = shrink_ratio
        self.out_height = int(out_height*shrink_ratio)
        self.out_width = int(max_ratio*out_height*shrink_ratio)
        self.channels = channels
        self._vis = False

        super(QuadROI, self).__init__()

    def generate_grid(self, quad_boxes, output_w, output_h):
        # 从quad 等距采样出对应的size: return torch (N,W,H,2)
        left_line = np.linspace(quad_boxes[0],quad_boxes[3],output_h)
        right_line = np.linspace(quad_boxes[1],quad_boxes[2],output_h)
        
        res = []
        for i in range(output_h):
            line = np.linspace(left_line[i], right_line[i], output_w)
            res.append(line)

        res = torch.FloatTensor(res)
        return res

    def roi(self, img, img_quad_boxes) : # -> rec后的image:torch.Tensor

        out_height = self.out_height
        out_width = self.out_width
        channels = self.channels

        res_images = []
        #masks = []
        img = img.unsqueeze(0)
        h, w = np.shape(img)[2:4]
        ratio = out_height / h

        for i, quad_box in enumerate(img_quad_boxes):
            
            test_grid = self.generate_grid(quad_box, output_h=out_height, output_w=out_width)
            test_grid = test_grid.to(img.device)

            test_grid[:,:,0] = (test_grid[:,:,0]*2/w) -1
            test_grid[:,:,1] = (test_grid[:,:,1]*2/h) -1
            rec_x = F.grid_sample(img, test_grid.unsqueeze(0), mode='bilinear')
            res_images.append(rec_x.squeeze(0))

            if self._vis:
                x_vis = rec_x.squeeze(0).permute(1,2,0).cpu().numpy()*255

                image = img.squeeze(0).permute(1,2,0).cpu().numpy()*255
                cv2.imwrite("/home/wx/tmp_pic/x_vis.jpg",x_vis)
                cv2.imwrite("/home/wx/tmp_pic/image_vis.jpg",image)
                #print(rec_x.shape)

        return res_images

    def forward(self, x_batch, boxes_batch):

        shrink_ratio = self.shrink_ratio
        rec_x = []
        batch_sizes = []

        for i, img_quad_boxes in enumerate(boxes_batch):

            x = x_batch[i]
            batch_sizes.append(len(img_quad_boxes))
            img_quad_boxes *= shrink_ratio
            rec_x += self.roi(x, img_quad_boxes)

        return torch.stack(rec_x), batch_sizes

    def vis(self, new_img_torch, id_name):
        img = new_img_torch.numpy().transpose(1,2,0)*255
        cv2.imwrite("/home/wx/tmp_pic/_"+ str(id_name)+'.jpg', img )

if __name__ == "__main__":
    pass



