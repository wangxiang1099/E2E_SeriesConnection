import os,math,sys,yaml
import numpy as np
from tqdm import tqdm
from easydict import EasyDict

import cv2
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader,Dataset

from datasets.dataset.pretrain_invoice.pretrainInvoice import pretrainInvoiceDataset
from datasets.transform import *

from evalVis.tensorboard import TensorWriter
from evalVis.eval.iou import jaccard

EVAL_CNT = 0
best_v = -1

def load_model(pretrain_path, model, device):
        
    states = torch.load(pretrain_path, map_location=device)
    #states = transform_dict(states)
    model.load_state_dict(states, strict=True)
    print("load_model <<", pretrain_path)
    return model

def my_collate_fn(batch):
    
    names = [x[0] for x in batch]
    images = [x[1] for x in batch]

    detect_gt = [x[2]['detect_gt'] for x in batch]
    image_vis = [x[2]['image_vis'] for x in batch]
    detect_gt_border = [x[2]['detect_gt_border'] for x in batch]
    detect_border_mask = [x[2]['detect_border_mask'] for x in batch]

    boxes_batch = [x[2]['boxes'] for x in batch]

    rec_texts = [x[3] for x in batch]

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
    
    dataset = pretrainInvoiceDataset(root=cfg.root,
        s1_targets_preprocess = prepocess_detect_label(),
        s2_targets_preprocess = None,
        image_encoding_transforms = get_image_encoding_transforms()
        )
    # split the dataset in train and test set
    torch.manual_seed(1)

    indices = torch.randperm(len(dataset)).tolist()
    datasetTrain = torch.utils.data.Subset(dataset, indices[:-cfg.val_size])
    datasetVal = torch.utils.data.Subset(dataset, indices[-cfg.val_size:])

    print(len(datasetTrain))
    print(len(datasetVal))

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
    
    if cfg.optimizer.name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr,
                            betas=(cfg.optimizer.beta1, 0.999), weight_decay=0.00002)

    return optimizer, None
    
def test_training(params):
    pass

def train_one_epoch(cfg, epoch, train_loader, val_loader, device, model, optimizer, writer):
    
    model.train()
    for i_batch, (names, images, detect_targets, rec_targets) in enumerate(train_loader):
		
        images = images.to(device)
        detect_targets['detect_gt'] = detect_targets['detect_gt'].to(device)
        detect_targets['detect_gt_border'] = detect_targets['detect_gt_border'].to(device)
        detect_targets['detect_border_mask'] = detect_targets['detect_border_mask'].to(device)
        
        try:
            detect_loss, recog_loss = model(images, s1_target = detect_targets, 
                                                s2_target = rec_targets)

        except Exception as e:
            print("wrong in " +str(i_batch))
            continue
        
        if torch.isnan(recog_loss) or torch.isnan(detect_loss):
            print("wrong in" +str(i_batch))
            continue
        
        loss = detect_loss + 10*recog_loss

        # model.zero_grad()
        # recog_loss.backward(retain_graph = True)
        # if i_batch == 0:
        #     print("grad",model.share_conv.in2.weight.grad)

        #  optimizer.step()
        #  loss = recog_loss + detect_loss
        #loss.to(device)
        #if i_batch%10 == 0:

        loss.backward()
        optimizer.step()

        writer.update_loss(s1_loss=detect_loss.data)
        writer.update_loss(s2_loss=recog_loss.data)
        writer.update_loss(all_loss=loss.data)

        if (i_batch) % cfg.log_print_freq == 0:

            loss = writer.dump_loss("loss", i_batch + epoch*len(train_loader))

            print('[%d/%d][%d/%d] all_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['all_loss']))
            print('[%d/%d][%d/%d] s2_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['s2_loss']))
            print('[%d/%d][%d/%d] s1_loss: %f' %
                  (epoch, cfg.epochs, i_batch, len(train_loader), loss['s1_loss']))

        if (i_batch) % cfg.i_batch_eval_freq == 0:
            
            eval_block(cfg, model, train_loader, val_loader, device, writer)
            model.train()
            
def save_model(model, save_path):

    torch.save(model.state_dict(),save_path)
    print("save ok in "+ save_path)

def _8point4point( bbox):
    bbox = np.reshape(bbox, (4, 2)).astype(np.int32)
    x1 = np.min(bbox[:, 0])
    x2 = np.max(bbox[:, 0])
    y1 = np.min(bbox[:, 1])
    y2 = np.max(bbox[:, 1])
    return (x1, y1, x2, y2)

def eval_model(model, dataloader, device, max_cnt=100, vis_batch= 5):
    
    # detect_model_load
    # 评估模型， 最大计数为 max_cnt == 1000 实际上由于batch原因会超过这个数字
    model.eval()

    max_batch  = 1000
    batch_size = 1

    n_correct = 0
    item_cnt = 0
    acc = 0

    iou_v_list = []
    vis_pictures = {}

    with torch.no_grad():
        for i_batch, (names, images, detect_targets, rec_targets) in tqdm(enumerate(dataloader)):
                
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
                detect_res, recog_res = model(images, s1_target = detect_targets, 
                                                      s2_target = rec_targets)
            except Exception as e:
                print("wrong in "+ i_batch)
                continue

            vis_boxes_pic = [] 
            
          #  print(detect_res.keys())
            for i, boxes in enumerate(detect_res['boxes_batch']):
                
                empty_flag = False

                image_vis = detect_targets['image_vis'][i]
                if boxes == []:
                    empty_flag = True
                    boxes = [(0,0,1,1)]

                #print(boxes.shape)
               # print(detect_targets['boxes_batch'][i].shape)
                boxes = list(map(_8point4point, boxes))

                ttargets = detect_targets['boxes_batch'][i] 
                ttargets = list(map(_8point4point, ttargets))
                iou_matrix = jaccard(torch.FloatTensor(boxes), torch.FloatTensor(ttargets))

                boxes_pair_index = iou_matrix.max(0).indices

                iou_mean = iou_matrix.max(1).values.sum()/len(boxes)
                iou_v_list.append(iou_mean)
                
                for box in boxes:
                    cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

                #for box in boxes_batch[i]:
                #    cv2.rectangle(image_vis, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

                #cv2.imwrite("/home/wx/tmp_pic/test_vis_box.jpg",image_vis)
                image_vis = torch.Tensor(image_vis).permute(2,0,1)
                vis_boxes_pic.append(image_vis)
                
                """
                if empty_flag:
                    item_cnt += len(rec_targets['texts'][i])
                    continue
                """
                for i_text, target in enumerate(rec_targets['texts'][i]['texts']):
                    
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

    acc = n_correct/(item_cnt)
    mean_iou = np.mean(iou_v_list)

    return acc, mean_iou, vis_pictures

def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero

def eval_block(cfg, model, train_loader, val_loader, device, writer):
    
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

def main(train_cfg, dataset_cfg, eval_vis_cfg, path_cfg, model):

    device = torch.device("cuda:%d" %(train_cfg.device) if torch.cuda.is_available() else "cpu")
    
    # YAML_PATH = "/home/wx/project/E2E_SeriesConnection/exp/pretrainInvoice/pretrain.yaml"
    # cfg = setup(YAML_PATH)
    # make it 
    model = model.to(device)

    if train_cfg.load_pretrain_model:
        model = load_model(train_cfg.pretrain_model_path, model, device)

    train_cfg.pretrain_save_path = path_cfg.pretrain_save_path

    model._inference_using_bounding_box = True
    print('init_model setting ok!')

    train_loader, val_loader = dataloader_init(dataset_cfg)

    writer = TensorWriter(path_cfg.process_path, loss_items = ["s1_loss","s2_loss","all_loss"], 
            metric_items = eval_vis_cfg.metric_items)  

    optimizer, scheduler = train_init(train_cfg, model)

    model.register_backward_hook(backward_hook)
    #test_training()

    for epoch in range(0, train_cfg.epochs):

        train_one_epoch(train_cfg, epoch, train_loader, val_loader, device, model, optimizer, writer)
        
    writer.close()

if __name__ == "__main__":

    main()