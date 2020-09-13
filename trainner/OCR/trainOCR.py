import os,math,sys,yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
from collections import OrderedDict

sys.path.append("/home/wx/project/E2E_SeriesConnection")

import cv2

import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader,Dataset

from modelZoo.iOCR.end2end import End2EndOcr

from datasets.OCR.pretrain_invoice.pretrainInvoice import pretrainInvoiceDataset
from datasets.transform import *

from evalVis.tensorboard import TensorWriter
from evalVis.eval.iou import jaccard

EVAL_CNT = 0
best_v = -1

def setup(yaml_path):
    """
    Create configs and perform basic setups.
    """
    torch.manual_seed(1)
    #params = EasyDict()
    yaml_file = open(yaml_path)
    
    cfg = yaml_file.read()
    params = yaml.safe_load(cfg)
    params = EasyDict(params)

    pretrain_save_path = os.path.join(params.result_dir, params.expeiment_name,'pretrain')
    result_path = os.path.join(params.result_dir, params.expeiment_name,'result')
    process_path = os.path.join(params.result_dir, params.expeiment_name,'process_path')

    if not os.path.exists(pretrain_save_path):
        os.makedirs(pretrain_save_path)

    params['pretrain_save_path'] = pretrain_save_path

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    params['result_path'] = result_path
    
    if not os.path.exists(process_path):
        os.makedirs(process_path)
    params['process_path'] = process_path
    
    return params

def load_model(pretrain_path, model, device):
        
    states = torch.load(pretrain_path, map_location=device)
    model.load_state_dict(states, strict=False )
    print("load_model <<", pretrain_path)
    return model

def model_init(cfg, device, load_pretrain_model=True):
    
    model = End2EndOcr().to(device)

    if cfg.load_pretrain_model:
        model = load_model(cfg.pretrain_model_path, model, device)

    model._inference_using_bounding_box = True
    print('init_model setting ok!')

    return model

def my_collate_fn(batch):
    
    names = [x[0] for x in batch]
    images = [x[1] for x in batch]

    detect_gt = [x[2]['detect_gt'] for x in batch]
    image_vis = [x[2]['image_vis'] for x in batch]
    detect_gt_border = [x[2]['detect_gt_border'] for x in batch]
    detect_border_mask = [x[2]['detect_border_mask'] for x in batch]

    boxes_batch = [x[2]['horizonal_boxes'] for x in batch]

    rec_texts = [x[3]['texts'] for x in batch]

    images = torch.stack(images)
    detect_gt = torch.stack(detect_gt)
    detect_gt_border = torch.stack(detect_gt_border)
    detect_border_mask = torch.stack(detect_border_mask)

    detect_targets = {}
    detect_targets['detect_gt'] = detect_gt
    detect_targets['image_vis'] = image_vis
    detect_targets['detect_gt_border'] = detect_gt_border
    detect_targets['detect_border_mask'] = detect_border_mask
    detect_targets['boxes_batch'] = boxes_batch

    rec_targets = {}
    rec_targets['texts'] = rec_texts

    return names, images, detect_targets, rec_targets

def dataloader_init(cfg):
    
    dataset = pretrainInvoiceDataset(root='/home/wx/data/iocr_training/syn6000invoice',
        s1_targets_preprocess = prepocess_detect_label(),
        s2_targets_preprocess = None,
        image_encoding_transforms = get_image_encoding_transforms()
        )

    print(dataset[0])

    # split the dataset in train and test set
    torch.manual_seed(1)
 
    indices = torch.randperm(len(dataset)).tolist()
    datasetTrain = torch.utils.data.Subset(dataset, indices[:-1000])
    datasetVal = torch.utils.data.Subset(dataset, indices[-1000:])

    assert datasetTrain[0]
    assert datasetVal[0]
    
    train_loader = DataLoader(datasetTrain, batch_size=cfg.train_batch_size, shuffle=True, num_workers=4,
               collate_fn=my_collate_fn)

    val_loader = DataLoader(datasetVal, batch_size=cfg.val_batch_size, shuffle=True, num_workers=4,
               collate_fn=my_collate_fn)

    print('init data loader ok!')
    return train_loader, val_loader 

def train_init(cfg, model):
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]        
        
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                          betas=(cfg.beta1, 0.999), weight_decay=0.00002)

    return optimizer, None
    
def test_training(params):
    pass

def train_one_epoch(cfg, epoch, train_loader, val_loader, device, model, optimizer, writer):
    
    model.train()
    for i_batch, (names, images, detect_targets, rec_targets) in enumerate(train_loader):
		
        boxes = detect_targets["boxes_batch"]
        images = images.to(device)
        detect_targets['detect_gt'] = detect_targets['detect_gt'].to(device)
        detect_targets['detect_gt_border'] = detect_targets['detect_gt_border'].to(device)
        detect_targets['detect_border_mask'] = detect_targets['detect_border_mask'].to(device)

        try:
            detect_res, recog_res, detect_loss, recog_loss = model.forward(images, detect_target = detect_targets, 
                                                                               boxes_batch = boxes, 
                                                                               rec_target = rec_targets)
        except Exception as e:
            print("wrong in " +str(i_batch))
            continue
        
        if torch.isnan(recog_loss) or torch.isnan(detect_loss):
            print("NANNNNNNN.......  wrong in" +str(i_batch))
            continue
        
        model.zero_grad()
        loss = cfg.detect_loss_weight*detect_loss +  cfg.rec_loss_weight*recog_loss
        loss.backward()
        optimizer.step()

        writer.update_loss(detect_loss=detect_loss.data)
        writer.update_loss(rec_loss=recog_loss.data)
        writer.update_loss(all_loss=loss.data)

        if (i_batch) % cfg.log_print_freq == 0:

            loss = writer.dump_loss("loss", i_batch + epoch*len(train_loader))

            print('[%d/%d][%d/%d] all_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['all_loss']))
            print('[%d/%d][%d/%d] rec_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['rec_loss']))
            print('[%d/%d][%d/%d] detect_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['detect_loss']))

        if (i_batch) % cfg.i_batch_eval_freq == 0:
            
            eval_block(cfg, model, train_loader, val_loader, device, writer)
            model.train()
            
def save_model(model, save_path):

    torch.save(model.state_dict(),save_path)
    print("save ok in "+ save_path)

def eval_detect_block():
    pass

def eval_rec_block():
    pass


def eval_model(model, dataloader, device, max_cnt=100, vis_batch= 5):
    
    # detect_model_load
    # 评估模型， 最大计数为 max_cnt == 1000 实际上由于batch原因会超过这个数字
    model.eval()

    max_batch  = 1000

    n_correct = 0
    item_cnt = 0
    acc = 0

    iou_v_list = []
    vis_pictures = {}

    with torch.no_grad():
        for i_batch, (names, images, detect_targets, rec_targets) in tqdm(enumerate(dataloader)):
            
            boxes_batch = detect_targets["boxes_batch"]

            batch_size = len(images)
            #print(batch_size)
            if i_batch * batch_size > max_cnt:
                max_batch = i_batch
                break
        
            images = images.to(device)
            detect_targets['detect_gt'] = detect_targets['detect_gt'].to(device)
            detect_targets['detect_gt_border'] = detect_targets['detect_gt_border'].to(device)
            detect_targets['detect_border_mask'] = detect_targets['detect_border_mask'].to(device)
            
            try:
                detect_res, recog_res, detect_loss, recog_loss = model.forward(images, detect_target = detect_targets, 
                                                                                       boxes_batch = boxes_batch, 
                                                                                       rec_target = rec_targets)
            except Exception as e:
                continue

            vis_boxes_pic = [] 
            
            for i, boxes in enumerate(detect_res['boxes_batch']):

                empty_flag = False

                image_vis = detect_targets['image_vis'][i]
                if boxes == []:
                    empty_flag = True
                    boxes = [(0,0,1,1)]

                iou_matrix = jaccard(torch.FloatTensor(boxes), torch.FloatTensor(boxes_batch[i]))
                boxes_pair_index = iou_matrix.max(0).indices

                iou_mean = iou_matrix.max(1).values.sum()/len(boxes)
                iou_v_list.append(iou_mean)
                
                for box in boxes:
                    cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                for box in boxes_batch[i]:
                    cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                if i == 0:
                    cv2.imwrite("/home/wx/tmp_pic/" + str(EVAL_CNT)+ ".jpg", image_vis)
                
                image_vis = torch.Tensor(image_vis).permute(2,0,1)
                vis_boxes_pic.append(image_vis)
                
                """
                if empty_flag:
                    item_cnt += len(rec_targets['texts'][i])
                    continue
                """
                
                for i_text, target in enumerate(rec_targets['texts'][i]):
                    
                    #index = boxes_pair_index[i_text]
                    pred = recog_res[i][i_text]

                    if pred == target: 
                        n_correct += 1

                    if i_batch < 1:
                        print("pred",pred)
                        print("target",target)

                    item_cnt += 1

            vis_boxes_pic = torch.stack(vis_boxes_pic)
            # (2,3,1024,2048)            
            if i_batch < vis_batch:
                
                vis_pictures['segmentation'] = detect_res['segmentation']
                vis_pictures['binary_pred'] = detect_res['binary']
                vis_pictures['detect_gt'] = detect_targets['detect_gt']
                vis_pictures['boxes_pred'] = vis_boxes_pic

    acc = n_correct/item_cnt
    mean_iou = np.mean(iou_v_list)

    return acc, mean_iou, vis_pictures

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

def eval_block(cfg, model, train_loader,val_loader, device, writer):
    
        global EVAL_CNT
        global best_v

   # try:
        train_acc, train_mean_iou, train_vis_pictures = eval_model(model, train_loader, device)
        print("epoch_",EVAL_CNT ,"train_acc:",train_acc)
        print("epoch_",EVAL_CNT ,"train_mean_iou:",train_mean_iou)

        val_acc, val_mean_iou, val_vis_pictures = eval_model(model, val_loader, device)
        print("epoch_",EVAL_CNT ,"val_acc:",val_acc)
        print("epoch_",EVAL_CNT ,"val_mean_iou:",val_mean_iou)

        writer.update_metric(val_acc = val_acc)
        writer.update_metric(train_acc = train_acc)

        writer.update_metric(train_mean_iou = train_mean_iou)
        writer.update_metric(val_mean_iou = val_mean_iou)

        writer.dump_metric("metric", EVAL_CNT)

        for k, v in train_vis_pictures.items():
            print(v.shape)
            print(k)
            writer.add_images("train_"+k, v, EVAL_CNT)

        for k, v in val_vis_pictures.items():
            writer.add_images("val_"+k, v, EVAL_CNT)
        
        if val_acc > best_v:

            best_v = val_acc
            save_path = os.path.join(cfg.pretrain_save_path,
                                        'best.pth')

            with open(os.path.join(cfg.pretrain_save_path,"best.txt"), 'w') as f:

                f.write("val_acc: "+str(val_acc)+"\n")
                f.write("train_acc: "+str(train_acc)+"\n")
                f.write("train_mean_iou: "+str(train_mean_iou)+"\n")
                f.write("val_mean_iou: "+str(val_mean_iou)+"\n")

            save_model(model, save_path)

   # except Exception as e:
    #    print('wrong  eval !! continue Training')

        EVAL_CNT += 1

def main(yaml_path=None):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yaml_path = "/home/wx/project/E2E_SeriesConnection/config/expPretrainE2E.yaml"
    #YAML_PATH = yaml_path
    cfg = setup(yaml_path)
    device = torch.device("cuda:%d" %(cfg.device) if torch.cuda.is_available() else "cpu")
    # make it 
    model = model_init(cfg, device, load_pretrain_model=True)
    train_loader, val_loader = dataloader_init(cfg)
    writer = TensorWriter(cfg.process_path, loss_items = ["rec_loss","detect_loss","all_loss"], 
            metric_items = ['val_acc','train_acc','train_mean_iou','val_mean_iou'])  

    optimizer, scheduler = train_init(cfg, model)

    model.register_backward_hook(backward_hook)
    #test_training()

    for epoch in range(0, cfg.epochs):

        train_one_epoch(cfg, epoch, train_loader, val_loader, device, model, optimizer, writer)
        
    writer.close()

if __name__ == "__main__":

    main()
