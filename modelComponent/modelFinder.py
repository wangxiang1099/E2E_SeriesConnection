from modelZoo.model import *

class ModelFinder():

    def __init__(self):
        pass

    def __call__(self, build_params):

        # 工厂函数
        if build_params is None:
            build_params = {}

        if build_params['name'] == "DB":
            model = DB(build_params)

        if build_params['name'] == "CRNN":
            model = CRNN(build_params)
        
        self.test_model(model.test_data, model)

        return model

    def test_model(self, test_data, model):
        
        pass
        