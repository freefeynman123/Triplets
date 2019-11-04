import torch
import torch.nn.functional as F
from torch import nn


class TripletSoftLoss(nn.Module):
    """
    Defines loss with softmax values for embeddings distances.
    """

    def __init__(self):
        super(TripletSoftLoss, self).__init__()

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1).pow(0.5)
        distance_negative = (anchor - negative).pow(2).sum(1).pow(0.5)
        soft = nn.Softmax(dim=0)
        result = soft(torch.stack((distance_positive, distance_negative), dim=0))
        base = torch.stack((torch.zeros(result.shape[1]), torch.ones(result.shape[1])),
                           dim=0).to(result.device)
        return (result.squeeze() - base).pow(2).mean()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
