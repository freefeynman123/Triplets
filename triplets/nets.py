from typing import Tuple

import torch
from torch import nn


class TripletNet(nn.Module):
    def __init__(
            self,
            pretrained_net: nn.Module
    ) -> None:
        super(TripletNet, self).__init__()
        self.pretrained_net = pretrained_net

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            x3: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns embeddings of three images - potentially anchor, positive and negative example.
        :param x1: Tensor containing image data.
        :param x2: Tensor containing image data.
        :param x3: Tensor containing image data.
        :return: Embeddings of image data.
        """
        embedding1 = self.pretrained_net(x1)
        embedding2 = self.pretrained_net(x2)
        embedding3 = self.pretrained_net(x3)
        return embedding1, embedding2, embedding3
