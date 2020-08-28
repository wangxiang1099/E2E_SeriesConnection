from modelZoo.convLayer import *

class ShareConvFinder():

    def __init__(self):
        pass

    def __call__(self, model_name, build_params):

        # 工厂函数
        if build_config is None:
            build_config = {}

        if model_name == "resnet18_fpn":
            model = FPN(resnet18(), in_channels=[64,128,256,512])

        if model_name == "resnet50_fpn":
            model = FPN(resnet50())
        
        return model

