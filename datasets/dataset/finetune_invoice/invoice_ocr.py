import os,cv2
import json
from tqdm import tqdm
import glob
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch

MIX = 0
SCAN = 1
PIC = 2

class InvoiceDataset(Dataset):
    
    # 基类
    def __init__(self, root, mode=MIX, path=True, transforms=None):
        
        self.mode = mode
        self.root = root
        self.path = path
        self.transforms = transforms

        pic_path_list = glob.glob(os.path.join(self.root, 'picture', '*.jpg'))
        scan_path_list = glob.glob(os.path.join(self.root, 'scan', '*.jpg'))
                
        if self.mode == MIX:
            self.image_path_list = pic_path_list + scan_path_list
        elif self.mode == SCAN:
            self.image_path_list = scan_path_list
        elif self.mode == PIC:
            self.image_path_list = pic_path_list
        
    @staticmethod
    def read_label(path):
        #### checkcode words may be "unknown" if people can't recognize it from image !!!!!!!!!
        with open(path, 'r') as f:
            label = {}
            for line in f.readlines():
                no, key, box, text = line.split()
                no = int(no)
                box = list(map(int, box.split(',')))
                label[no] = {'key_name': key, 'boxes': box, 'texts': text}
            assert not (6 in label and 7 in label), 'label error in {}'.format(path)
            if 6 in label:
                assert len(label[6]['texts']) == 20 or label[6]['texts'] == 'unknown', 'label error in {}'.format(path)
            if 7 in label:
                assert len(label[7]['texts']) == 20 or label[7]['texts'] == 'unknown', 'label error in {}'.format(path)
            return label

    def get_all(self):
        data = {}
        for image_path in self.image_path_list:
            image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])
            label = self.read_label(os.path.join(image_dir, image_name + '.txt'))
            mode = os.path.basename(image_dir)
            idx = mode + '_' + image_name

            if not self.path:
                image = cv2.imread(image_path)
                data[idx] = {'img': image, 'keys': label}
            else:
                data[idx] = {'img': image_path, 'keys': label}
        return data


    def __len__(self):
        return len(self.image_path_list)
    
    def __getitem__(self, idx):

        image_path = self.image_path_list[idx]
        image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])
        target = self.read_label(os.path.join(image_dir, image_name + '.txt'))
        mode = os.path.basename(image_dir)
        name = mode + '_' + image_name

        if not self.path:
            image = cv2.imread(image_path)
        else:
            image = image_path

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return name, image, target


# 用于检测数据预处理转换
class End2EndInvoiceDataset(InvoiceDataset):

    def __init__(self, root, skip_type=[], mode=MIX, transforms_detect=None,
                 transforms_recognition=None, prepare=False):

        super(End2EndInvoiceDataset, self).__init__(root=root, 
                                                    mode=mode,
                                                    path=False,
                                                    transforms=None)
        
        self.transforms_detect = transforms_detect
        self.transforms_recognition = transforms_recognition
        # 数据转换
        self.skip_type = skip_type

    def __len__(self):

        return len(self.image_path_list)

    def __getitem__(self,idx):

        image_path = self.image_path_list[idx]
        image_dir, image_name = os.path.split(os.path.splitext(image_path)[0])
        target = self.read_label(os.path.join(image_dir, image_name + '.txt'))
        mode = os.path.basename(image_dir)
        name = mode + '_' + image_name

        image = cv2.imread(image_path)
        h,w,_ = image.shape
        #print(h,w, _)
        """
        detection part
        """
        targets = {"anno_label":{},"anno_labels_path":os.path.join(image_dir, image_name + '.txt'),
                   "origin_h": h, "origin_w":w}

        for k,v in target.items():
            target = {'ids':k,'boxes':v['boxes'],'texts':v['texts']}
            targets["anno_label"][k] = target

        if self.transforms_detect:
            image, targets = self.transforms_detect(image, targets)
        
        return name, image, targets    

# 用于端到端 双路 训练 检测和识别 以及检测训练
class End2EndWithLoadInvoiceDataset(InvoiceDataset):

    
    def __init__(self, root, skip_type=[], mode=MIX, transforms_detect=None,
                 transforms_recognition=None, prepare=False):

        path_txt = os.path.join(root, "full.txt")
        self.data = []

        with open(path_txt,'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(line.strip().split(" | "))
        self.transforms_detect = transforms_detect
        self.transforms_recognition = transforms_recognition
        # 数据转换
        self.skip_type = skip_type

    def __len__(self):

        return len(self.data)

    def __getitem__(self,idx):

        origin_image = self.data[idx][0]
        detect_mask_image = self.data[idx][1]
        detect_gt = self.data[idx][2]
        detect_gt_border = self.data[idx][3]
        detect_border_mask = self.data[idx][4]
        text_label = self.data[idx][5]

        name  = os.path.split(os.path.splitext(origin_image)[0])[1]
        image_origin = cv2.imread(origin_image)

        # detect_part
        h, w, _ = image_origin.shape
        
        targets = {"anno_labels":{},"origin_image":None,
                   "detect_mask":None,"detect_gt":None,
                   "detect_gt_border":None, "detect_border_mask":None,
                   "origin_height": -1,"origin_width": -1}

        f = open(text_label,'r',encoding='utf-8')
        labels = json.load(f)
        f.close()

        targets["anno_labels"] = labels
        targets["detect_mask"] = cv2.imread(detect_mask_image,0)
        targets["detect_gt"] = cv2.imread(detect_gt,0)
        targets["detect_gt_border"] = cv2.imread(detect_gt_border,0)
        targets["detect_border_mask"] = cv2.imread(detect_border_mask,0)
        targets["origin_image"] = image_origin.copy()
        targets["origin_height"] = h
        targets["origin_height"] = w

        if self.transforms_detect:
            image, targets = self.transforms_detect(image_origin, targets)
        
        #recognition part
        #rec_image_parts = []

        ids = []
        texts = []
        boxes = []

        for k,v in targets['anno_labels'].items():
            if v['texts'] == "unknown":
                continue

            bounding_box = v['boxes']

            boxes.append(bounding_box)
            ids.append(k)
            texts.append(v['texts'])

        rec_targets = {'ids':ids, 'texts':texts}
        
        if self.transforms_recognition:
            boxes, rec_targets = self.transforms_recognition(boxes, rec_targets)  
        
        return name, image, targets, boxes, rec_targets



if __name__ == '__main__':
    #dataset = InvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=MIX)

    #for i, (name, image, target) in enumerate(iter(dataset)):
    #    print(i, name, image, target)

    dataset = End2EndInvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=MIX)
    print(dataset[0])    
    #dataset = RecognitionInvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=MIX)

    #for i, (name, image, target) in enumerate(iter(dataset)):
    #    print(i, name, image, target)


   # prepare_data_info()
    """
    from transform import get_transform_detect
    dataset = End2EndWithLoadInvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=0,
                    transforms_detect=get_transform_detect())
    
    print(dataset[0][2]['detect_mask'].min())
    """