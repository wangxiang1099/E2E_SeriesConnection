
import cv2 
import numpy as np
import torch
from collections import OrderedDict
import json

def main():

    pass

def transform_dict(state_dict):

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if "thresh" in k or "binarize" in k:
            name = "detect_branch."+ k[21:]
        else:
            if "backbone" in k:
                name = "share_conv.backbone."+k[22:]
            else:
                name = "share_conv."+k[21:]
        new_state_dict[name] = v 
    return new_state_dict
    
def prepare_data_info(root='/home/wx/data/iocr_training/invoices'):
    
    dataset = End2EndInvoiceDataset(root=root, mode=0)

    with open(os.path.join(root,"full.txt"),'w') as f:
        
        for i, (name, image, targets) in enumerate(iter(dataset)):
            
            base_dir = os.path.join(root,'transform_to_train')

            origin_path = os.path.join(base_dir,'origin', name+".jpg")
            detect_mask_path = os.path.join(base_dir,'detect_mask', name+".png")
            detect_gt_path = os.path.join(base_dir,'detect_gt', name+".png")
            detect_gt_border_path = os.path.join(base_dir,'detect_gt_border', name+".png")
            detect_border_mask_path = os.path.join(base_dir,'detect_border_mask', name+".png")
            anno_labels_path = os.path.join(base_dir,'anno_labels', name+".json")
            
            path_all = [origin_path, detect_mask_path, detect_gt_path,
                        detect_gt_border_path, detect_border_mask_path, anno_labels_path]

            flag = True
            for p in path_all:
                if not os.path.isfile(p):
                    print("wrong in ", p)
                    flag = False
                    break
                    
            if flag:
                line = " | ".join(path_all)
                f.write(line)
                f.write("\n")

                
if __name__ == "__main__":

    output_dir = "/home/wx/data/iocr_training/invoices/transform_to_train"
    resize_height = 1<<10
    resize_width = 1<<11

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    path = "/home/wx/pretrain_model/detection/DB/pre-trained-model-synthtext-resnet18"
    #path = "/home/wx/pretrain_model/detection/DB/final"
    states = torch.load(path, map_location=device)
    states = transform_dict(states)
    
    import sys, os
    sys.path.append("/home/wx/project/now_project/iocr_trainning")
    
    from datasets.invoices.invoice_ocr import End2EndInvoiceDataset
    from datasets.invoices.transform import get_prepared_transform_detect
    from model_zoo.ocr.end2end import End2EndOcr 

    from vis.detect_representation import SegDetectorRepresenter
    from torch.utils.data import DataLoader

    
    #print(res.keys())
    net = End2EndOcr().to(device)
    net.load_state_dict(states, strict = False)
    net.eval()
    net.only_run_detect()

    dataset = End2EndInvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=0,
                transforms_detect=get_prepared_transform_detect(
                resize_shape = (resize_width,resize_height)))
    
    represent = SegDetectorRepresenter()
    vis = True
    save = True

    for i, (name, image, targets) in enumerate(iter(dataset)):
        
        image = image.to(device).unsqueeze(0)
        print(name)
        print("doing ______ %d / %d"%(i, len(dataset)))

        #print(targets.keys())
        #print(targets['anno_label'])

        detect_res, recog_res, _,_ = net(image, targets=targets)

        segmentation = represent.represent_segmentation(detect_res['binary'], (resize_height, resize_width))
        gt_dilate = targets['gt_dilate']
        # make mask 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
        segmentation_vis = segmentation[0].permute(1,2,0).cpu().data.numpy()

        #print(segmentation_vis.shape)
        segmentation_dilate = abs(1- cv2.dilate(np.uint8(segmentation_vis), kernel, iterations=5))

        #print(detect_res['binary'][0].)
        #pred_vis = detect_res['binary'][0].permute(1,2,0)*255
        mask = segmentation_dilate + gt_dilate
        # detect_gt make
        if save:
            # image_origin
            image_origin_path = os.path.join(output_dir, "origin",name+".jpg")
            cv2.imwrite(image_origin_path, targets['origin'])
            # detect_mask
            detect_mask_path = os.path.join(output_dir, "detect_mask",name+".png")
            cv2.imwrite(detect_mask_path, mask*255)
            # detect_gt_border
            detect_gt_border_path = os.path.join(output_dir, "detect_gt_border",name+".png")
            cv2.imwrite(detect_gt_border_path, targets['detect_gt_border']*255)
            # detect_gt
            detect_gt_path = os.path.join(output_dir, "detect_gt",name+".png")
            cv2.imwrite(detect_gt_path,targets['detect_gt']*255)
            # detect_mask
            detect_border_mask_path = os.path.join(output_dir, "detect_border_mask",name+".png")
            cv2.imwrite(detect_border_mask_path, targets['detect_border_mask']*255)
            # anno_label json
            anno_path = os.path.join(output_dir, "anno_labels",name+".json")
            
            with open(anno_path,'w',encoding='utf-8') as f:
                f.write(json.dumps(targets['anno_label'], ensure_ascii=False))

            #cv2.imwrite('/home/wx/tmp_pic/mask_test.jpg', mask*255)

        if i == 5:
            continue
    
    prepare_data_info()