from torch import nn


class TripletNet(nn.Module):
    def __init__(self, pretrained_net):
        super(TripletNet, self).__init__()
        self.pretrained_net = pretrained_net

    def forward(self, x1, x2, x3):
        embedding1 = self.pretrained_net(x1)
        embedding2 = self.pretrained_net(x2)
        embedding3 = self.pretrained_net(x3)
        return embedding1, embedding2, embedding3
