
from modelZoo.model import *

class ModelFinder():

    def __init__(self):
        pass

    def __call__(self, model_name, build_params):

        # 工厂函数
        if build_config is None:
            build_config = {}

        if model_name == "DB":
            model = DB(build_params)

        if model_name == "CRNN":
            model = CRNN(build_params)
        
        self.test_model(model.test_data, model)

        return model

    def test_model(self, test_data, model):
        
        pass
        