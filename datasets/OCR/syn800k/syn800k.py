import torch
import torch.utils.data as data
import scipy.io as scio
import re
import itertools
import random
from PIL import Image
import torchvision.transforms as transforms
import time
import os ,cv2
import numpy as np

class Synth80k(data.Dataset):

    def __init__(self, synthtext_folder, viz=False, transforms=None, image_augs=None, preprocess_detect_label=None):

        super(Synth80k, self).__init__()
        self.synthtext_folder = synthtext_folder
        gt = scio.loadmat(os.path.join(synthtext_folder, 'gt.mat'))
        self.wordbox = gt['wordBB'][0]
        self.image_names = gt['imnames'][0]
        self.imgtxt = gt['txt'][0]

        self.transforms = transforms        
        self.viz = viz
        self.image_augs = image_augs
        self.preprocess_detect_label = preprocess_detect_label

    def __getitem__(self, index):
        
        image_name = self.image_names[index]
        image, word_bboxes, txt_target = self.load(index)

        if self.image_augs is not None:
            image, word_bboxes = self.image_augs(image, word_bboxes)
        
        detect_target = {}
        if self.preprocess_detect_label is not None:
            _, detect_target = self.preprocess_detect_label(image, word_bboxes)

        if self.transforms is not None:
            image, _ = self.transforms(image, None)            

        if self.viz:
            cv2.imwrite("/home/wx/tmp_pic/gt_syn.jpg", detect_target['detect_gt'].permute(1,2,0).numpy()*255)
            cv2.imwrite("/home/wx/tmp_pic/image_syn.jpg", image.permute(1,2,0).numpy()*255)
        
        return image_name, image, detect_target, word_bboxes, txt_target

    def __len__(self):
        return len(self.imgtxt)

    def load(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框 polygons 输出
        '''
        img_path = os.path.join(self.synthtext_folder, self.image_names[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.wordbox[index].ndim == 2:
            self.wordbox[index] = np.expand_dims(self.wordbox[index],axis = 2)
        
        word_bboxes = self.wordbox[index].transpose((2, 1, 0))

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        #character_bboxes = []
        #total = 0
        return image, word_bboxes, words


def generate_grid(quad_boxes, output_w, output_h):
    # 从quad 等距采样出对应的size: return torch (N,W,H,2)
    left_line = np.linspace(quad_boxes[0],quad_boxes[3],output_h)
    right_line = np.linspace(quad_boxes[1],quad_boxes[2],output_h)
    
    res = []
    for i in range(output_h):
        line = np.linspace(left_line[i], right_line[i], output_w)
        res.append(line)

    res = torch.FloatTensor(res)
    return res

if __name__ == '__main__':

    import torch.nn.functional as F
    from torchvision.transforms import functional as Ft
    from prepocess_detect_label import prepocess_detect_label
    from image_augs import image_augs
    from transform import get_transforms_detect

    synthtextloader = Synth80k('/home/wx/data/iocr_training/Syn800k/SynthText',image_augs=image_augs(),
                                preprocess_detect_label = prepocess_detect_label(),
                                transforms=get_transforms_detect())
                                
    image_name, image, detect_target, word_bboxes, txt_target = synthtextloader[0]

    print(word_bboxes[0])
    test_grid = generate_grid(word_bboxes[0], output_h=32, output_w=32*8)

    print(image.shape)
    test_grid[:,:,0] = (test_grid[:,:,0]*2/image.shape[2]) -1
    test_grid[:,:,1] = (test_grid[:,:,1]*2/image.shape[1]) -1
    x = F.grid_sample(image.unsqueeze(0), test_grid.unsqueeze(0), mode='bilinear')

    x_vis = x.squeeze(0).permute(1,2,0).numpy()*255

    image = image.permute(1,2,0).numpy()*255
    cv2.imwrite("/home/wx/tmp_pic/x_vis.jpg",x_vis)
    cv2.imwrite("/home/wx/tmp_pic/image_vis.jpg",image)
    print(x.shape)


