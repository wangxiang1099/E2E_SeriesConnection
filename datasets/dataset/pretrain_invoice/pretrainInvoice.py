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
    def __init__(self, root, split="data_full", transforms=None):
        
        self.root = root
        self.transforms = transforms

        self.data = []
        data_path = os.path.join(root, split+".txt")
        self.data += self.read_path(data_path)  
        self.transforms = transforms
                
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
        
        with open(label_path, 'r', encoding='utf-8') as f:

            loc_ret = []
            txt_ret = []

            for line in f.readlines():
                line = line.strip()
                line = line.split(" ")
                assert len(line) == 2

                loc_label = line[0]
                txt_label = line[1]

                loc_label = tuple(map(int,loc_label.split(",")))
                
                loc_ret.append(loc_label)
                txt_ret.append(txt_label)

        return loc_ret, txt_ret

    def preprocess(self, image, boxes):
        
        shrink_ratio = 0.4
        ret = {}
        h,w, _ = image.shape
        gt = np.zeros((1, h, w), dtype=np.float32)

        for box in boxes:
            
            x1,y1, x2,y2 = box

            area = int(x2-x1)*int(y2-y1)
            longth = 2*int(x2-x1) + 2*int(y2-y1)

            distance = area * \
                    (1 - np.power(shrink_ratio, 2)) / longth
            
            
            shrinked_x1 = x1 + distance/2
            shrinked_y1 = y1 + distance/2
            shrinked_x2 = x2 - distance/2
            shrinked_y2 = y2 - distance/2
            
            #shrinked_x1 = x1
            #shrinked_y1 = y1
            #shrinked_x2 = x2
            #shrinked_y2 = y2

            shrinked = np.array(((shrinked_x1,shrinked_y1),(shrinked_x2,shrinked_y1),
                        (shrinked_x2,shrinked_y2),(shrinked_x1,shrinked_y2)))

            cv2.fillPoly(gt[0,:,:], [shrinked.astype(np.int32)], 1)

        ret['gt'] = torch.FloatTensor(gt)
        ret['image_vis'] = image.copy()
            
        return ret

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx, vis =True):

        data_idx = self.data[idx]
        image_path = data_idx['image_path']
        label_path = data_idx['label_path']

        image = cv2.imread(image_path)
        image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])

        boxes, txt_target = self.read_label(label_path)
        detect_target = self.preprocess(image,boxes)

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        if vis:
            cv2.imwrite("/home/wx/tmp_pic/gt_syn.jpg", detect_target['gt'].permute(1,2,0).numpy()*255)
            cv2.imwrite("/home/wx/tmp_pic/image_syn.jpg", image.permute(1,2,0).numpy()*255)
            
        return image_name, image, detect_target, boxes, txt_target

if __name__ == '__main__':
    from transform import get_transforms_recognition
    dataset = pretrainInvoiceDataset(root='/home/wx/data/iocr_training/syn6000invoice',transforms=get_transforms_recognition())
    print(dataset[0])  
