class Trainer:

    def __init__(self, config, model, dataset, evalVis):
        pass

    def setup(self):
        pass

    def load_model(self, pretrain_path, model, device):
        
        states = torch.load(pretrain_path, map_location=device)
        #states = transform_dict(states)
        model.load_state_dict(states, strict=True)
        print("load_model <<", pretrain_path)
        return model

    def model_init(self,cfg, device, load_pretrain_model=True):
        
        model = End2EndOcr().to(device)

        if cfg.load_pretrain_model:
            model = load_model(cfg.pretrain_model_path, model, device)

        model._inference_using_bounding_box = True
        print('init_model setting ok!')

        return model

    def my_collate_fn(self,batch):
    
        names = [x[0] for x in batch]
        images = [x[1] for x in batch]

        detect_gt = [x[2]['detect_gt'] for x in batch]
        image_vis = [x[2]['origin_image'] for x in batch]

        boxes = [x[3] for x in batch]

        rec_texts = [x[4]['texts'] for x in batch]

        images = torch.stack(images)
        detect_gt = torch.stack(detect_gt)

        detect_targets = {}
        detect_targets['detect_gt'] = detect_gt
        detect_targets['image_vis'] = image_vis

        rec_targets =  {}
        rec_targets['texts'] = rec_texts

        return names, images, detect_targets, boxes, rec_targets

    def dataloader_init(self,cfg):
    
        dataset = End2EndWithLoadInvoiceDataset(root='/home/wx/data/iocr_training/invoices', mode=0,
                        transforms_detect=get_transform_detect(),
                        transforms_recognition=get_transforms_recognition())

        # split the dataset in train and test set
        torch.manual_seed(1)

        indices = torch.randperm(len(dataset)).tolist()
        datasetTrain = torch.utils.data.Subset(dataset, indices[:-300])
        datasetVal = torch.utils.data.Subset(dataset, indices[-300:])

        assert datasetTrain[0]
        assert datasetVal[0]
        
        train_loader = DataLoader(datasetTrain, batch_size=cfg.train_batch_size, shuffle=False, num_workers=0,
                collate_fn=my_collate_fn)

        val_loader = DataLoader(datasetVal, batch_size=cfg.val_batch_size, shuffle=False, num_workers=0,
                collate_fn=my_collate_fn)

        print('init data loader ok!')
        return train_loader, val_loader 
    
    def train_init(self,cfg, model):
        
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]        
            
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                            betas=(cfg.beta1, 0.999), weight_decay=0.00002)

        return optimizer, None
        
    def test_training(params):
        pass

    def train_one_epoch(cfg, epoch, train_loader, val_loader, device, model, optimizer, writer):
    
        model.train()
        for i_batch, (names, images, detect_targets, boxes, rec_targets) in enumerate(train_loader):
            
            images = images.to(device)
            detect_targets['detect_gt'] = detect_targets['detect_gt'].to(device)
            
            try:
                detect_res, recog_res, detect_loss, recog_loss = model.forward(images, detect_target = detect_targets, 
                                                                                    boxes_batch = boxes, 
                                                                                    rec_target = rec_targets)
            except Exception as e:
                print("wrong in " +str(i_batch))
                continue
            
            if torch.isnan(recog_loss) or torch.isnan(detect_loss):
                print("wrong in" +str(i_batch))
                continue
            
            loss = detect_loss +  recog_loss
            model.zero_grad()
            #recog_loss.backward(retain_graph = True)

            #optimizer.step()
            loss = recog_loss + detect_loss
            #loss.to(device)
            if i_batch%1 == 0:

                loss.backward()
                if i_batch == 0:
                    print("grad",model.share_conv.in2.weight.grad)
                
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

    def backward_hook(self, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0   # replace all nan/inf in gradients to zero


    def train(self,cfg):

        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
        YAML_PATH = "/home/wx/project/now_project/iocr_trainning/trainingExperiments/OCR/end2end_ocr/exp_refine_end2end/fine_tune.yaml"
        #YAML_PATH = args.YAML_PATH
        cfg = setup(YAML_PATH)
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


        

    

