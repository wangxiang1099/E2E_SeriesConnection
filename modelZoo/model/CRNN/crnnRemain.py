import torch.nn as nn
import torch.nn.functional as F
import torch


class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN_remain(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH=32, nc=1, nclass=11, n_rnn=2, leakyRelu=False):
        super(CRNN_remain, self).__init__()
        assert imgH % 8 == 0, 'imgH has to be a multiple of 8'

        ks = [3, 3, 3, 3, 3, 3]
        ps = [1, 1, 1, 1, 1, 1]
        ss = [1, 1, 1, 1, 1, 1]
        nm = [64, 64, 128, 128, 256, 256]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=True):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        convRelu(1)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2,1),(2,1)))  # 64x16x64
        convRelu(2)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2,1),(2,1)))  # 128x8x32
        convRelu(4)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(2),
                        nn.MaxPool2d((2, 2), (2, 1)))  # 256x4x16

        self.cnn = cnn
        self.rnn = nn.Linear(256, nclass, bias=True)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, nh, nh),
        #     BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('---forward propagation---')

        #print("input",input.size())
        conv = self.cnn(input)
        b, c, h, w = conv.size()
       # print("output",conv.size())
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]

       # print("rnn_front",conv.size())
        output = F.log_softmax(self.rnn(conv), dim=2)
        
       # print(self.rnn(conv).size())
        return output

def test():
    net = CRNN(imgH =8, nc=256, nclass =11)
    fms = net(torch.randn(20,256,8,80))
    print(fms.size())

if __name__ == "__main__":
    test()
