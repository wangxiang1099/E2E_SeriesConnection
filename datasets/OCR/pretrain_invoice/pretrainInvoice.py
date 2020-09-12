import os,cv2
import json
import os
import json
from tqdm import tqdm
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

class pretrainInvoiceDataset(Dataset):
    
    # 基类
    def __init__(self, root, split="data_full", image_augs=None, s1_targets_preprocess=None,
                                                s2_targets_preprocess=None,
                                                image_encoding_transforms = None):
        
        self.root = root

        self.data = []
        data_path = os.path.join(root, split+".txt")
        self.data += self.read_path(data_path)  

        self.image_augs = image_augs
        self.s1_targets_preprocess = s1_targets_preprocess
        self.s2_targets_preprocess = s2_targets_preprocess
        self.image_encoding_transforms = image_encoding_transforms
                
    def read_path(self, data_path):
        # read data path

        with open(data_path, 'r', encoding='utf-8') as f:

            ret = []
            for line in f.readlines():
                line = line.strip()
                line = line.split(" ")
                assert len(line) == 2

                image_path = line[0]
                label_path = line[1]
                if os.path.exists(image_path) and os.path.exists(label_path):
                    ret.append({"image_path": image_path,
                                "label_path": label_path})
                                
        return ret
    
    def read_label(self,label_path):
        
        label_res = {"quad_boxes":None, "horizonal_boxes":None, "texts":None}
        with open(label_path, 'r', encoding='utf-8') as f:

            quad_loc_ret = []
            txt_ret = []
            loc_ret = []

            for line in f.readlines():
                line = line.strip()
                line = line.split(" ")
                assert len(line) == 2

                loc_label = line[0]
                txt_label = line[1]

                loc_label = tuple(map(int,loc_label.split(",")))

                x1, y1, x2, y2 = loc_label[0],loc_label[1],loc_label[2],loc_label[3]

                quad_loc_label = ((x1, y1), 
                             (x1, y2),
                             (x2, y2),
                             (x2, y1))
                
                loc_ret.append(loc_label)
                quad_loc_ret.append(quad_loc_label)
                txt_ret.append(txt_label)

            label_res['horizonal_boxes'] = np.array(loc_ret)
            label_res['quad_boxes'] = np.array(quad_loc_ret, dtype=np.float64)
            label_res['texts'] = txt_ret

        return label_res

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, vis =False):

        data_idx = self.data[idx]

        image_path = data_idx['image_path']
        label_path = data_idx['label_path']

        image = cv2.imread(image_path)
        image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])

        targets = self.read_label(label_path)

        # 处理images 和labels 特征 
        if self.image_augs is not None:
            image, targets = self.image_augs(image, targets)
        
        s1_targets = {"quad_boxes": targets['quad_boxes'],
                      "horizonal_boxes": targets['horizonal_boxes']}

        s2_targets = {"texts": targets['texts']}

        if self.s1_targets_preprocess is not None:
            _, s1_targets = self.s1_targets_preprocess(image, s1_targets)

        if self.s2_targets_preprocess is not None:
            _, s2_targets = self.s2_targets_preprocess(image, s2_targets)

        if self.image_encoding_transforms is not None:
            image = self.image_encoding_transforms(image)            

        if vis:
            cv2.imwrite("/home/wx/tmp_pic/image_syn.jpg", image.permute(1,2,0).numpy()*255)

        return image_name, image, s1_targets, s2_targets

if __name__ == '__main__':

    import sys
    sys.path.append("/home/wx/project/E2E_SeriesConnection")
    from datasets.transform import *
    
    dataset = pretrainInvoiceDataset(root='/home/wx/data/iocr_training/syn6000invoice',
        s1_targets_preprocess = prepocess_detect_label(),
        s2_targets_preprocess = None,
        image_encoding_transforms = get_image_encoding_transforms()
        )
    
    print(dataset[0])  
