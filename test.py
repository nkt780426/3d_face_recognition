import torch

# Giả sử output của mô hình (softmax) có kích thước (batch_size, num_classes)
# Batch size là 4, với 3 lớp (0, 1, 2)
x_pose = torch.tensor([
    [0.1, 0.8, 0.1],  # Sample 1: Lớp 1 có xác suất cao nhất
    [0.2, 0.3, 0.5],  # Sample 2: Lớp 2 có xác suất cao nhất
    [0.7, 0.2, 0.1],  # Sample 3: Lớp 0 có xác suất cao nhất
    [0.3, 0.4, 0.3]   # Sample 4: Lớp 1 có xác suất cao nhất
])

pose = torch.tensor([
    1, 1, 1, 1
])
# Sử dụng torch.max để tìm lớp có xác suất lớn nhất (along dimension 1 - theo chiều lớp)
probs, predicted_classes = torch.max(x_pose, dim=1)
preds = torch.gather(x_pose, 1, pose.unsqueeze(1)).squeeze(1)


# In kết quả
print("Xác suất lớn nhất:", probs)
print("Lớp dự đoán có xác suất lớn nhất:", predicted_classes)
print("Lớp có label đúng:", preds)
