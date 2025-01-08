import torch
import torch.nn.functional as F
from torch.nn import Parameter
import math
from .backbone.iresnet import *
from .TripletFaceRecognition import EmbeddingNet

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MagLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, easy_margin=True, l_margin=0.45, u_margin=0.8, l_a=10, u_a=110):
        super(MagLinear, self).__init__()
        # Tạo ra ma trận cần train của lớp FullyConnected này
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        # Can thiệp vào quá trình khởi tạo trọng số: 
        # uniform_(-1,1): đảm bảo phân phối đồng đều (uniform) trong khoảng [-1,1]
        # renorm_(2, 1, 1e-5): chuẩn hóa lại các giá trị theo 1 chiều cụ thể. 
        #   2: Chuẩn hóa theo chuẩn L2(Euclidean norm).
        #   1: Thực hiện chuẩn hóa trên chiều thứ nhất của tensor (các hàng của ma trận self.weight)
        #   1e-5: Giá trị ngưỡng (epsilon) để tránh chia cho số 0
        # Khi dùng chuẩn hóa này, mỗi hàng của ma trận trọng số sẽ được điều chỉnh sao cho norm của chúng không vượt quá một giá trị cụ thể (giới hạn bởi epsilon).
        # Bản thân mỗi cột của ma trận này là ma trận weight của 1 neutron hay vector tâm của class identity.
        self.weight.data.uniform_(-1,1).renorm_(2, 1, 1e-5).mul_(1e5)
        
        self.easy_margin = easy_margin
        self.l_margin, self.u_margin = l_margin, u_margin
        self.l_a, self.u_a = l_a, u_a
    
    # Generate adaptive margin
    def _margin(self, x):
        margin = (self.u_margin-self.l_margin) / \
            (self.u_a-self.l_a)*(x-self.l_a) + self.l_margin
        return margin
    
    # x là batch_embedding
    def forward(self, x):
        # Tính độ dài của batch embedding sinh ra batch ai
        # Dim = 1: Tính độ dài theo chiều batch của tensor vào. Keepdim=True đảm bảo tensor sau tính toán sẽ có shape (batch_size,1)
        # clamp(min, max) giới hạn giá trị trong tensor sao cho tất cả các phần tử nhỏ hơn min sẽ được thay thế bằng min, và tất cả các phần tử lớn hơn max sẽ được thay thế bằng max.
        x_norm = torch.norm(x, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        
        # ada chắc là viết tắt của adaptive
        ada_margin = self._margin(x_norm)
        
        cos_m, sin_m = torch.cos(ada_margin), torch.sin(ada_margin)
        
        # Tính theta yi
        # dim bằng 0 tức là chuẩn hóa ma trận theo cột => mỗi cột là ma trận trọng số của 1 neutron như đã nói và sẽ có độ dài là 1
        weight_norm = F.normalize(self.weight, dim=0)
        # cosine similarity (sự tương đồng cosin). Kết quả thu được tensor chứa batch_cosin_theta
        cos_theta = torch.mm(F.normalize(x), weight_norm)
        # Đảm bảo giá trị cosin của các vector không vượt quá giới hạn => Thừa
        cos_theta = cos_theta.clamp(-1, 1)
        # sin sẽ luôn dương
        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        # cos(a+b) = cos(a)*cos(b) - sin(a)*sin(b)
        cos_theta_m = cos_theta * cos_m - sin_theta * sin_m

        # easy_margin = True để điều chỉnh cos_theta_m, nó sẽ bằng công thức trên nếu cos_theta>0 và không điều chỉnh gì nếu cos_theta<0
        # Cách này làm đơn giản việc tính toán
        # Nếu muốn điều chỉnh thì cần thêm vào threshold nào đó (code của họ để mặc định easy_margin=False)
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            mm = torch.sin(math.pi - ada_margin) * ada_margin
            threshold = torch.cos(math.pi - ada_margin)
            cos_theta_m = torch.where(cos_theta > threshold, cos_theta_m, cos_theta - mm)
            
        # Trả về các giá trị này để tính loss và accuracy
        # [cos_theta, cos_theta_m] là logits của lớp này. Lý do thêm x_norm để phục vụ MagFace+ => Train không ?
        # cos_theta chính là accuracy, nó đo cosine similarity giữa vector tâm và feature vector.
        return [cos_theta, cos_theta_m], x_norm
    
class MagFaceRecognition(torch.nn.Module):
    def __init__(self, conf):
        super(MagFaceRecognition, self).__init__()
        if conf['backbone'] == 'iresnet18':
            backbone = iresnet18(num_classes=conf['embedding_size'])
        if conf['backbone'] == 'iresnet34':
            backbone = iresnet34(num_classes=conf['embedding_size'])
        if conf['backbone'] == 'iresnet50':
            backbone = iresnet50(num_classes=conf['embedding_size'])
        if conf['backbone'] == 'iresnet101':
            backbone = iresnet101(num_classes=conf['embedding_size'])
        if conf['backbone'] == 'iresnet152':
            backbone = iresnet152(num_classes=conf['embedding_size'])
        if conf['backbone'] == 'inception-resnet-v1':
            backbone = EmbeddingNet(len_embedding=conf['embedding_size'])
        self.feature_extractor = backbone
        self.classification = MagLinear(conf['embedding_size'], conf['num_class'])
    
    def forward(self, x):
        x = self.feature_extractor(x)
        logits, x_norm = self.classification(x)
        
        return logits, x_norm
    
    def get_embedding(self, x):
        return self.feature_extractor(x)
    
class MagFaceConcatRecognition(torch.nn.Module):
    def __init__(self, conf):
        super(MagFaceConcatRecognition, self).__init__()
        self.albedo_extractor = MagFaceRecognition(conf)
        self.depthmap_extractor = MagFaceRecognition(conf)
        self.normalmap_extractor = MagFaceRecognition(conf)
        
        albedo_checkpoint = torch.load(conf['albedo_checkpoint'], weights_only=True)
        self.albedo_extractor.load_state_dict(albedo_checkpoint)
        depthmap_checkpoint = torch.load(conf['depthmap_checkpoint'], weights_only=True)
        self.depthmap_extractor.load_state_dict(depthmap_checkpoint)
        normalmap_checkpoint = torch.load(conf['normalmap_checkpoint'], weights_only=True)
        self.normalmap_extractor.load_state_dict(normalmap_checkpoint)
        
        self.fc = torch.nn.Linear(3 * conf['embedding_size'], conf['embedding_size'], bias=False)
        self.classification = MagLinear(conf['embedding_size'], conf['num_class'])

    def forward(self, x):
        assert x.size(1) == 3, "Expected 3 inputs: albedo, depthmap, normalmap"
        
        albedo = x[:, 0, :, :, :]
        depthmap = x[:, 1, :, :, :]
        normalmap = x[:, 2, :, :, :]
        
        albedo_feature = self.albedo_extractor.get_embedding(albedo)
        depthmap_feature = self.depthmap_extractor.get_embedding(depthmap)
        normalmap_feature = self.normalmap_extractor.get_embedding(normalmap)
        
        concatenated_feature = torch.cat((albedo_feature, depthmap_feature, normalmap_feature), dim=1)  # Dim 1 = feature dimension
        
        x = self.fc(concatenated_feature)
        
        logits, x_norm = self.classification(x)
        
        return logits, x_norm
    
    def get_embedding(self, x):
        assert x.size(1) == 3, "Expected 3 inputs: albedo, depthmap, normalmap"
        
        albedo = x[:, 0, :, :, :]
        depthmap = x[:, 1, :, :, :]
        normalmap = x[:, 2, :, :, :]
        
        albedo_feature = self.albedo_extractor.get_embedding(albedo)
        depthmap_feature = self.depthmap_extractor.get_embedding(depthmap)
        normalmap_feature = self.normalmap_extractor.get_embedding(normalmap)
        
        concatenated_feature = torch.cat((albedo_feature, depthmap_feature, normalmap_feature), dim=1)  # Dim 1 = feature dimension
        
        x = self.fc(concatenated_feature)
        
        return x