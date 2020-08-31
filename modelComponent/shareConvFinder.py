from modelZoo.convLayer import *

class ShareConvFinder():

    def __init__(self):
        pass

    def __call__(self, build_params):

        # 工厂函数
        if build_params is None:
            build_params = {}

        if build_params['name'] == "resnet18_fpn":
            model = FPN(resnet18(), in_channels=[64,128,256,512])

        if build_params['name'] == "resnet50_fpn":
            model = FPN(resnet50())
        
        return model

