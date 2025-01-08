import torch
import torch.nn.functional as F

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MagLoss(torch.nn.Module):
    def __init__(self, conf):
        super(MagLoss, self).__init__()
        self.l_a = conf['l_a']
        self.u_a = conf['u_a']
        self.scale = conf['scale']
        self.lambda_g = conf['lambda_g']

    def calc_loss_G(self, x_norm):
        g = 1/(self.u_a**2) * x_norm + 1/(x_norm)
        # x_norm là 1 tensor chứa batch các ai. Do đó g là 1 tensor chứa batch các giá trị g thỏa mãn cho mỗi sample.
        # Việc lấy giá trị trung bình giúp làm mượt quá trình huấn luyện (nếu không loss thay đổi rất nhanh, vì mỗi mẫu trong batch có thể có độ khó khác nhau). Khi tính trung bình, các giá trị sẽ được làm mượt, giúp quá trình huấn luyện diễn ra mượt mà hơn và không bị ảnh hưởng quá nhiều bởi các mẫu cực trị.
        # Đảm bảo rằng loss không thay đổi theo kích thước batch. Điều này rất quan trọng trong việc huấn luyện mô hình một cách công bằng giữa các batch có kích thước khác nhau.
        return torch.mean(g)

    # input: là logits thu được của layer MagLinear.
    # target: là tensor label thực tế của cả batch
    # x_norm: là logits thu được của layer MagLinear
    def forward(self, logits, target, x_norm):
        loss_g = self.calc_loss_G(x_norm)

        cos_theta, cos_theta_m = logits
        
        cos_theta = self.scale * cos_theta
        cos_theta_m = self.scale * cos_theta_m
        
        # Khởi tạo 1 tensor chứa các giá trị 0 có kích thước giống như cos_theta
        one_hot = torch.zeros_like(cos_theta)
        # Onehotcoding label. Phương thức scatter_ sẽ thay các giá trị của one_hot tại chỉ mục được xác định bởi target thành 1.
        # 1: dim=1, Chỉ thị cho PyTorch biết sẽ thay đổi giá trị dọc theo chiều thứ nhất (theo các cột, tức là các lớp).
        # target.view(-1, 1): chuyển target thành tensor cột có kích thước (batch_size,1)
        # 1.0: giá trị được gán
        one_hot.scatter_(1, target.view(-1, 1), 1.0)
        
        # Onehot là ma trận mục tiêu, ta muốn áp dụng cos_theta_m cho lớp mục tiêu này và giữ nguyên giá trị cosin_theta cho các lớp khác.
        # Đây là cách thức áp dụng margin vào cosine similarity để tăng cường phân biệt giữa lớp mục tiêu và các lớp khác.
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        loss_id = F.cross_entropy(output, target, reduction='mean')
        # Chỉ trả về các thành phần tính loss chứ chưa thực sự tính
        loss_g = self.lambda_g * loss_g
        return loss_id , loss_g