import torch
import torch.nn.functional as F
import pandas as pd

class FocalLoss(torch.nn.Module):
    
    def __init__(self, file_path, label_name, gamma_weights=None):
        super(FocalLoss, self).__init__()
        self.gamma_weights = gamma_weights or {}
        
        # Tính alpha cho từng lớp
        self.alpha = self._compute_alpha(file_path, label_name)

        # Chuyển gamma_weights thành tensor
        self.gamma = torch.tensor(
            [self.gamma_weights.get(i, 1.0) for i in range(len(self.gamma_weights))],
            dtype=torch.float32
        )


    def _compute_alpha(self, file_path, label_name):
        # Đọc dữ liệu từ file CSV
        self.data = pd.read_csv(file_path)
        
        # Đếm số lượng mẫu của từng nhãn
        label_counts = self.data[label_name].value_counts().sort_index()
        
        # Tổng số mẫu
        total_samples = label_counts.sum()
        
        # Tính tần suất ngược
        alpha = 1 - (label_counts.values / total_samples)
        
        # Chuẩn hóa alpha về khoảng [0, 1]
        alpha = (alpha - alpha.min()) / (alpha.max() - alpha.min() + 1e-8)
        
        # Giới hạn alpha trong khoảng [0.2, 0.8]
        min_alpha, max_alpha = 0.2, 0.8
        alpha = alpha * (max_alpha - min_alpha) + min_alpha
        
        # Chuyển alpha thành tensor
        return torch.tensor(alpha, dtype=torch.float32)



    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        y_true = y_true.long()
        y_pred_correct_class = y_pred.gather(1, y_true.unsqueeze(1)).squeeze(1)
        
        # Lấy alpha cho từng nhãn
        alpha = self.alpha.to(y_true.device)[y_true]  # Kích thước: (batch_size,)
        
        # Lấy gamma cho từng nhãn
        gamma = self.gamma.to(y_true.device)[y_true]  # Kích thước: (batch_size,)
        
        # Tính Focal Loss
        focal_weight = alpha * (1 - y_pred_correct_class) ** gamma
        log_prob = torch.log(y_pred_correct_class + 1e-8)  # Thêm epsilon để tránh log(0)
        focal_loss = -focal_weight * log_prob

        # Trả về trung bình của Focal Loss trong batch
        return focal_loss.mean()
    
if __name__ == '__main__':
    # batch_size = 4
    # num_classes = 3
    # y_pred = torch.tensor([[2.0, 1.0, 0.1],  # Logits cho mẫu 1
    #                     [0.5, 2.5, 0.3],  # Logits cho mẫu 2
    #                     [1.0, 0.2, 3.0],  # Logits cho mẫu 3
    #                     [0.3, 0.7, 1.5]]) # Logits cho mẫu 4
    # y_true = torch.tensor([0, 1, 2, 1])  # Nhãn thực của batch

    file_path = '../../../Dataset/train_set.csv'
    # gamma_weights = {0: 2.0, 1: 2.0, 2: 2.0}

    # focal_loss = FocalLoss(file_path=file_path, label_name="Pose", gamma_weights=gamma_weights)

    # # Tính Focal Loss
    # loss = focal_loss(y_pred, y_true)
    # print(f"Focal Loss: {loss.item()}")
    
    focal_loss = FocalLoss(file_path=file_path, label_name="id")
    print(focal_loss.alpha)