

class E2EModelBuild:

    def __init__(self, taskConfig):
        
        self.modelS1 = None
        self.modelS2 = None
        self.taskConfig = None
        self.MergeModel = None
        
    # build model
    def buildModel(self, taskConfig):
        
        S1 = "DB"
        S2 = "CRNN"
        modelS1 = ModelFinder(S1)
        modelS2 = ModelFinder(S2)

        # 模型裁剪 
        modelS1 = ClipperModel(self.modelS1)
        modelS2 = ClipperModel(self.modelS2)

        # 共享卷积
        shareConv = ShareConvFinder("resnet18_fpn")
        # 连接器配置 
        connector = ConnectorFinder("simple_roi")
        # loss配置 
        expandS2 = ExpandS2()

        self.E2Emodel = E2EModel(taskConfig, shareConv, modelS1, modelS2, connector, expandS2)
    
    def buildDataset(self, taskConfig):
        self.dataset = MergeDataset()

    def buildEvalVis(self, taskConfig):
        self.evalVis = EvalVis()

    def buildTrainer(self, taskConfig):

        loss = LossBuilder()    
        # 优化器、训练配置 
        self.trainer = Trainer(taskConfig, self.E2Emodel, self.dataset, self.evalVis, loss)
        # 子任务数据拓展
        # 假数据测试
        self._test_train()
        return # model class
    # test train
    def _test_train(self):
        return True

    # train model
    def train(self):

        res  = {
            "name":None,
            "status":None,
            "eval":None,
            "vis_pic_path":None,
            "train_model_path":None,
            "config_path":None
        }

        return res


if __name__ == "__main__":
    # 读配置 传参
    
    e2e = E2EModelBuild()
    e2e.buildModel()
    e2e.buildDataset()
    e2e.buildEvalVis()
    e2e.buildTrainer()
    e2e.train()

    print("OKOK!!!")

    








