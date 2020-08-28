from modelZoo.connector import *

class ConnectFinder():

    def __init__(self):
        pass

    def __call__(self, model_name, build_params):

        # 工厂函数
        if build_config is None:
            build_config = {}

        if model_name == "simple_roi":
            model = DB(build_params)

        if model_name == "roi":
            model = CRNN(build_params)
        
        self.test_model(model.test_data, model)

        return model

    def test_model(self, test_data, model):
        
        pass