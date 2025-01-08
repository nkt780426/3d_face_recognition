import torch
import torch.nn.functional as F

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class TripletLoss(torch.nn.Module):
    def __init__(self, margin=1.):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()