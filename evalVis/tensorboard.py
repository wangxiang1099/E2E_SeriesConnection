from tensorboardX import SummaryWriter
import copy
import torchvision
import torch

class Summary():
    '''TensorSummary: calculate mean values for params in one epoch, here params are dynamicly defined
    
        Args: 
            opt: parsed options from cmd or .yml(in config/ folder)
            
    '''
    def __init__(self):
        self.params = {}
        self.num = {}

    def register_params(self,*args):
        # dynamic register params for summary
        for arg in args:
            if not isinstance(arg, str):
                raise ValueError("parameter names should be string.")
            self.params[arg] = 0
            self.num[arg] = 0
        print("current summary have {}".format(self.params.keys()))
    
    def clear(self):
        # clear diction for new summary
        self.params = {}
        self.num = {}

    def reset(self):
        # reset all values to zero
        for key in self.params.keys():
            self.params[key] = 0
        
        for key in self.num.keys():
            self.num[key] = 0

    def update(self, **kwargs):
        # update params for one batch

        # sanity check
        for key in kwargs.keys():
            if key not in self.params:
                raise ValueError("Value Error : param {} not in summary diction".format(key))

        for (key, val) in kwargs.items():

            self.params[key] += val
            self.num[key] += 1

        return True

    def summary(self, is_reset=True, is_clear=False):
        # get mean value for all param data
        for (key, value) in self.params.items():
            value = value / self.num[key] if self.num[key] != 0 else 0
            self.params[key] = value

        # deep copy  
        mean_val = copy.deepcopy(self.params)

        # check is_reset and is_clear
        if is_reset:
            #print('before_reset',self.params, self.num)
            self.reset()
            #print("reset", self.params, self.num)
        if is_clear:
            self.clear()
        print('summary val',mean_val)
        # return mean value
        return mean_val

##############################################################

class MetricSummary(Summary):
    '''MetricSummary: calculate mean value for metrics'''
    def __init__(self, params):
        super(MetricSummary, self).__init__()
        self.register_params(*params)

class LossSummary(Summary):
    '''LossSummary: calculate mean value for loss'''
    def __init__(self, params):
        super(LossSummary, self).__init__()
        self.register_params(*params)


class TensorWriter(SummaryWriter):
    '''TensorWriter: numeric value visualization or image visualization inherit from SummaryWriter
    '''
    def __init__(self, dump_folder, loss_items, metric_items):

        super(TensorWriter, self).__init__(dump_folder)
        self.loss_summary = LossSummary(loss_items)
        self.metric_summary = MetricSummary(metric_items)

    def reset(self):
        self.loss_summary.reset()
        self.metric_summary.reset()

    def update_loss(self, **kwargs):
        self.loss_summary.update(**kwargs)
    
    def dump_loss(self,name,epoch):
        value = self.loss_summary.summary()
        self.add_scalars(name, value, epoch)
        return value

    def update_metric(self, **kwargs):
        self.metric_summary.update(**kwargs)
    
    def dump_metric(self,name,epoch):

        value = self.metric_summary.summary()
        self.add_scalars(name,value,epoch)
        return value

    def add_images(self, name, tensors, epoch):
        
        #print(tensors)
        tensors = self._to_cpu(tensors)
        # 加一下可以展示的代码
        grid = torchvision.utils.make_grid(tensors, nrow=1)
        self.add_image(name, grid, epoch)

    def _to_cpu(self, data):
        if isinstance(data, torch.autograd.Variable):
            data = data.data
        if isinstance(data, torch.cuda.FloatTensor):
            data = data.cpu()
        return data

if __name__ == "__main__":
    
    writer = TensorWriter('/home/wx/dump_results/iocr_training/only_rec_06_18/result',
                          loss_items = ['loss','loss2'],
                          metric_items = ['acc'])

    writer.update_loss(loss = 1)
    writer.update_loss(loss = 2)
    writer.update_loss(loss2 = 2)
    a = writer.dump_loss('loss3',1)
    print(a)
    writer.update_metric(acc = 0.9)
    writer.update_loss(loss = 2)
    writer.update_loss(loss = 3)
    a = writer.dump_loss('loss3',1)
    print(a)
    b = writer.dump_metric('acc',1)
    print(b)