from collections import OrderedDict

import torch
import torch.nn as nn
BatchNorm2d = nn.BatchNorm2d

class DBSegDetector(nn.Module):
    def __init__(self,inner_channels=256, k=50,
                 bias=False, adaptive=True, smooth=False, serial=False,
                 *args, **kwargs):
        '''
        bias: Whether conv layers have bias or not.
        adaptive: Whether to use adaptive threshold training or not.
        smooth: If true, use bilinear instead of deconv.
        serial: If true, thresh prediction will combine segmentation result as input.
        '''
        super(DBSegDetector, self).__init__()
        self.k = k
        self.serial = serial
        
        self.binarize = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, inner_channels//4, 2, 2),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels//4, 1, 2, 2),
            nn.Sigmoid())
        self.binarize.apply(self.weights_init)

        self.adaptive = adaptive
        if adaptive:
            self.thresh = self._init_thresh(
                    inner_channels, serial=serial, smooth=smooth, bias=bias)
            self.thresh.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def _init_thresh(self, inner_channels,
                     serial=False, smooth=False, bias=False):
        in_channels = inner_channels
        if serial:
            in_channels += 1
        self.thresh = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels //
                      4, 3, padding=1, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, inner_channels//4, smooth=smooth, bias=bias),
            BatchNorm2d(inner_channels//4),
            nn.ReLU(inplace=True),
            self._init_upsample(inner_channels // 4, 1, smooth=smooth, bias=bias),
            nn.Sigmoid())
        return self.thresh

    def _init_upsample(self,
                       in_channels, out_channels,
                       smooth=False, bias=False):
        if smooth:
            inter_out_channels = out_channels
            if out_channels == 1:
                inter_out_channels = in_channels
            module_list = [
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels, inter_out_channels, 3, 1, 1, bias=bias)]
            if out_channels == 1:
                module_list.append(
                    nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, padding=1, bias=True))

            return nn.Sequential(module_list)
        else:
            return nn.ConvTranspose2d(in_channels, out_channels, 2, 2)

    def forward(self, conv_map, gt=None, masks=None, training=False):

        # this is the pred module, not binarization module; 
        # We do not correct the name due to the trained model.
        binary = self.binarize(conv_map)
        result = OrderedDict(binary=binary)
        
        if not self.training:
            return result

        if self.adaptive and self.training:
            if self.serial:
                conv_map = torch.cat(
                        (conv_map, nn.functional.interpolate(
                            binary, conv_map.shape[2:])), 1)
            thresh = self.thresh(conv_map)
            thresh_binary = self.step_function(binary, thresh)
            result.update(thresh=thresh, thresh_binary=thresh_binary)
        
        return result

    def step_function(self, x, y):
        return torch.reciprocal(1 + torch.exp(-self.k * (x - y)))


if __name__ == "__main__":
    
    net = DBSegDetector()
    net.eval()
    fms = net(Variable(torch.randn(10, 256, 192, 192)))
    print(fms['binary'].size())

    