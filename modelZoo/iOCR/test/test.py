import torch
import torch.nn.functional as F
from torchvision.transforms import functional as F2
import cv2
import sys 
import numpy as np 
sys.path.append("/home/wx/project/now_project/iocr_trainning")
    
from datasets.invoices.invoice_ocr import End2EndInvoiceDataset

from torch.utils.data import DataLoader

dataset = End2EndInvoiceDataset(
        root='/home/wx/data/iocr_training/invoices', 
        #transforms_detect=get_transform_detect(),
        #transforms_recognition=get_transforms_recognition()
)
train_loader = DataLoader(dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=1,
                          collate_fn=lambda batch: tuple(zip(*batch)))

for i_batch, (name, image, target) in enumerate(train_loader):
    print("name",name)
    print("image", image)
    print('targets', target)
    input()
name, image, targets = dataset[0]

image_torch = F2.to_tensor(image).unsqueeze(0)


for t_each in targets:
    box = t_each['boxes']
    box_h = box[3] - box[1]

    ratio =  32/box_h

    x1,y1, x2, y2 = box[0], box[1], box[2], box[3]

    tx = box[0]*2 / image.shape[1]
    ty = box[1]*2 / image.shape[0]

    max_x = int((x2 - x1) * ratio)

    theta = torch.tensor([[1,0,tx],[0, 1,ty]]).unsqueeze(0)
    resize_h = int(image.shape[0]*ratio)
    resize_w = int(image.shape[1]*ratio)

    grid = F.affine_grid(theta, [1,3,resize_h,resize_w])
    output = F.grid_sample(image_torch, grid,mode='bilinear')

    new_img_torch = output[0][:,:32,:320]
    new_img_torch[:,:32,max_x:] = 0
    
    img = new_img_torch.numpy().transpose(1,2,0)*255
    cv2.imwrite("/home/wx/tmp_pic/_"+ str(t_each['ids'])+'.jpg', img )

