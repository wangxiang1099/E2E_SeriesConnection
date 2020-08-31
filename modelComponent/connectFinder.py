from modelZoo.connector import *

class ConnectFinder():

    def __init__(self):
        pass

    def __call__(self, build_params):

        # 工厂函数
        if build_params is None:
            build_params = {}

        if build_params['name'] == "SimpleROI":

            model = SimpleROI()
        
        if build_params['name'] == "QuadROI":

            model = QuadROI(shrink_ratio=0.25,out_height=32, max_ratio=12, channels=256)
 
        #self.test_model(model.test_data, model)

        return model

    def test_model(self, test_data, model):
        
        pass