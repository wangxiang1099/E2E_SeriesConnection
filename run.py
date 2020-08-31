from modelComponent import *
from trainner import *
import os 
from easydict import EasyDict
import yaml

class RUN:

    def __init__(self, taskConfigPath):
        
        self.config = self.setup(taskConfigPath)

    def setup(self, yaml_path):

        """
        Create configs and perform basic setups.
        """
        #params = EasyDict()
        yaml_file = open(yaml_path)
        
        cfg = yaml_file.read()
        params = yaml.safe_load(cfg)
        params = EasyDict(params)

        pretrain_save_path = os.path.join(params.PATH.result_dir, params.PATH.expeiment_name,'pretrain')
        result_path = os.path.join(params.PATH.result_dir, params.PATH.expeiment_name,'result')
        process_path = os.path.join(params.PATH.result_dir, params.PATH.expeiment_name,'process_path')

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

    # build model
    def buildModel(self):
        

        modelFinder = ModelFinder()
        modelS1 = modelFinder(self.config.S1)
        modelS2 = modelFinder(self.config.S2)

        # 模型裁剪 
        modelS1 = modelClipper(modelS1)
        modelS2 = modelClipper(modelS2)

        # 共享卷积
        shareConvFinder = ShareConvFinder()
        shareConv = shareConvFinder(self.config.ShareConv)

        # 连接器配置 
        connectFinder = ConnectFinder()
        connector = connectFinder(self.config.Connect)
        # loss配置 
        expandS2 = ExpandS2()

        self.E2Emodel = E2EModel(self.config.E2E, shareConv, modelS1, modelS2, connector, expandS2)
    
    def training_exp(self):

        train_main(self.config.Train, self.config.Dataset, self.config.EvalVis, self.config, self.E2Emodel)
    
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
    
    taskConfig = "/home/wx/project/E2E_SeriesConnection/TaskConfig.yaml"
    o = RUN(taskConfig)
    o.buildModel()
    o.training_exp()
    print("OKOK!!!")

    








