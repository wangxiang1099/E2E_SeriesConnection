from .tensorborad import TensorWriter
import os
import torch
import tqdm.tqdm

class EvalVisBase():

    def __init__(self, config):

        self.writer = TensorWriter(config.process_path, loss_items = ["s1_loss","s2_loss","all_loss"], 
            metric_items = ['val_acc','train_acc','train_mean_iou','val_mean_iou'])  

        self.evalList = []
        self.writer = None

        self.EVAL_CNT = 0
        self.best_v = 0

    def __del__(self):
        self.writer.close()
        
    def write_loss(self):
        pass

    def dump_loss(self):
        pass

    def save_model(self ,model, save_path):
        torch.save(model.state_dict(),save_path)
        print("save ok in "+ save_path)

    def eval_s1_res(self, s1_res, s1_targets):
        return {}

    def eval_s2_res(self, s2_res, s2_targets):
        return {}

    def eval_model(self, model, dataloader, device, max_cnt=100, vis_batch= 5):

        eval_dict = {"acc": None,
                     "mean_iou": None}

        eval_pic = {}
        
        with torch.no_grad():
            for i_batch, (names, images, s1_targets, s2_targets) in tqdm(enumerate(dataloader)):
                    
                batch_size = len(images)
                #print(batch_size)
                if i_batch * batch_size > max_cnt:
                    max_batch = i_batch
                    break

                images = images.to(device)
                
                try:
                    s1_res, s2_res = model(images, s1_targets=s1_targets, s2_targets=s2_targets)

                except Exception as e:
                    continue

                s1_eval_dict, s1_eval_pic = self.eval_s1_res(s1_res, s1_targets)
                s2_eval_dict, s2_eval_pic = self.eval_s1_res(s1_res, s1_targets)

        return s1_eval_dict, s1_eval_pic, s2_eval_dict, s2_eval_pic


    def eval_block(self, cfg, model,train_loader,val_loader, device, writer):
    
        train_acc, train_mean_iou, train_vis_pictures = self.eval_model(model, train_loader, device)
        print("epoch_",self.EVAL_CNT ,"train_acc:",train_acc)
        print("epoch_",self.EVAL_CNT ,"train_mean_iou:",train_mean_iou)

        val_acc, val_mean_iou, val_vis_pictures = self.eval_model(model, val_loader, device)
        print("epoch_",self.EVAL_CNT ,"val_acc:", val_acc)
        print("epoch_",self.EVAL_CNT ,"val_mean_iou:", val_mean_iou)

        writer.update_metric(val_acc = val_acc)
        writer.update_metric(train_acc = train_acc)

        writer.update_metric(train_mean_iou = train_mean_iou)
        writer.update_metric(val_mean_iou = val_mean_iou)

        writer.dump_metric("metric", self.EVAL_CNT)

        for k, v in train_vis_pictures.items():
            print(v.shape)
            print(k)
            writer.add_images("train_"+k, v, self.EVAL_CNT)

        for k, v in val_vis_pictures.items():
            writer.add_images("val_"+k, v, self.EVAL_CNT)
        
        if val_acc > self.best_v:

            self.best_v = val_acc
            save_path = os.path.join(cfg.pretrain_save_path,
                                        'best.pth')

            with open(cfg.pretrain_save_path, 'w') as f:

                f.write("val_acc: "+str(val_acc)+"\n")
                f.write("train_acc: "+str(train_acc)+"\n")
                f.write("train_mean_iou: "+str(train_mean_iou)+"\n")
                f.write("val_mean_iou: "+str(val_mean_iou)+"\n")

            self.save_model(model, save_path)

   # except Exception as e:
    #    print('wrong  eval !! continue Training')

        self.EVAL_CNT += 1



    

                