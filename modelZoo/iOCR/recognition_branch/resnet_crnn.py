import torch.nn as nn
import torch.nn.functional as F

from basenet.resnet50 import deformable_resnet50

# no use
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

class Resnet_CRNN(nn.Module):
    #                   32    1   37     256
    def __init__(self, imgH, nc, nclass, n_rnn=2, leakyRelu=False):
        
        super(Resnet_CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = deformable_resnet50()
        self.rnn = nn.Linear(2048, nclass, bias=True)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(512, nh, nh),
        #     BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('---forward propagation---')
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        output = F.log_softmax(self.rnn(conv), dim=2)
        return output


def test():

    net = Resnet_CRNN(imgH =32, nc=3, nclass =11).cuda()
    net.eval()
    inn = Variable(torch.randn((20,3,32,13)).cuda())
    print(inn.size())
    fms = net(inn)
    print(fms.size())

if __name__ == "__main__":
    
    test()


