from modelComponent import *
from datasets import *
from evalVis import *
from trainner import *

class E2EModelBuild:

    def __init__(self, taskConfig):
        
        self.taskConfig = None
        
    # build model
    def buildModel(self, config):
        
        S1 = "DB"
        S2 = "CRNN"

        modelFinder = ModelFinder()
        modelS1 = modelFinder(S1, config.S1)
        modelS2 = modelFinder(S2, config.S2)

        # 模型裁剪 
        modelS1 = modelClipper(modelS1)
        modelS2 = modelClipper(modelS2)

        # 共享卷积
        shareConvFinder = ShareConvFinder()
        shareConv = shareConvFinder("resnet18_fpn", config.shareConv)

        # 连接器配置 
        connectFinder = ConnectFinder()
        connector = connectFinder("simple_roi", config.connect)
        # loss配置 
        expandS2 = ExpandS2()

        self.E2Emodel = E2EModel(config.E2E, shareConv, modelS1, modelS2, connector, expandS2)
    
    def buildDataset(self, config):
        self.dataset = MergeDataset(config.dataset)

    def buildEvalVis(self, config):
        self.evalVis = EvalVis(config.evalVis)

    def buildTrainer(self, config):

        # 优化器、训练配置 
        self.trainer = Trainer(config.train, self.E2Emodel, self.dataset, self.evalVis)
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
    
    taskConfig = None
    e2e = E2EModelBuild(taskConfig)
    e2e.buildModel()
    e2e.buildDataset()
    e2e.buildEvalVis()
    e2e.buildTrainer()
    e2e.train()

    print("OKOK!!!")

    








