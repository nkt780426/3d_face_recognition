import torch

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def train_id_accuracy(predict, target):
    with torch.no_grad():
        _, pred = predict.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        correct_1 = correct[:1].contiguous().view(-1).float().sum(0, keepdim=True).item()
        correct_5 = correct[:5].contiguous().view(-1).float().sum(0, keepdim=True).item()
        return correct_1, correct_5