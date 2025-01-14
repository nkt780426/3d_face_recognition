import torch

from .focalloss.IdFocalLoss import IdFocalLoss
from .focalloss.FocalLoss import FocalLoss
from .MagLoss import MagLoss

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

class MultiTaskLoss(torch.nn.Module):
    
    def __init__(self, metadata_path:str, loss_weight:dict):
        super(MultiTaskLoss, self).__init__()
        
        self.id_loss = IdFocalLoss(metadata_path)
        # 0: female (244), 1: male (2630)
        self.gender_loss = FocalLoss(metadata_path, 'Gender', {0:0, 1:2})
        # 0: không đeo kính (2077), 1: đeo kính (797)
        self.spectacles_loss = FocalLoss(metadata_path, 'Spectacles',{0:1, 1:0})
        # 0: không râu (2014), 1: có râu (890)
        self.facial_hair_loss = FocalLoss(metadata_path, 'Facial_Hair', {0:1, 1:0})
        # 0: nhìn trực diện (2471), 1: nhìn nghiêng 1 chút (326), 2: lệch 30-45 độ (77)
        self.pose_loss = FocalLoss(metadata_path, 'Pose', {0:2, 1:0.25, 2:0})
        # 0: tóc che mặt (13), 1: tay che mặt (46), 2: không bị che khuất (2615)
        self.occlusion_loss = FocalLoss(metadata_path, 'Occlusion', {0:0, 1:0.1, 2:2})
        # 0: nhìn trực diện (2209), 1: tích cực (416), 2: các cảm xúc khác
        self.emotion_loss = FocalLoss(metadata_path, 'Emotion',{0:2, 1:0.25, 2:0})
        
        # hyper parameter
        self.spectacles_weight = loss_weight['loss_spectacles_weight']
        self.da_spectacles_weight = loss_weight['loss_da_spectacles_weight']
        self.occlusion_weight = loss_weight['loss_occlusion_weight']
        self.da_occlusion_weight = loss_weight['loss_da_occlusion_weight']
        self.facial_hair_weight = loss_weight['loss_facial_hair_weight']
        self.da_facial_hair_weight = loss_weight['loss_da_facial_hair_weight']
        self.pose_weight = loss_weight['loss_pose_weight']
        self.da_pose_weight = loss_weight['loss_da_pose_weight']
        self.gender_weight = loss_weight['loss_gender_weight']
        self.da_gender_weight = loss_weight['loss_da_gender_weight']
        self.emotion_weight = loss_weight['loss_emotion_weight']
        self.da_emotion_weight = loss_weight['loss_da_emotion_weight']
        
        
    def forward(self, logits, y):
        (
            (x_spectacles, x_da_spectacles), 
            (x_occlusion, x_da_occlusion),
            (x_facial_hair, x_da_facial_hair),
            (x_pose, x_da_pose),
            (x_emotion, x_da_emotion),
            (x_gender, x_da_gender),
            x_id
        ) = logits
        
        id, gender, spectacles, facial_hair, pose, occlusion, emotion = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6]
        
        loss_spectacles = self.spectacles_loss(x_spectacles, spectacles)
        loss_da_spectacles = self.spectacles_loss(x_da_spectacles, spectacles)
        
        loss_occlusion = self.occlusion_loss(x_occlusion, occlusion)
        loss_da_occlusion = self.occlusion_loss(x_da_occlusion, occlusion)
        
        loss_facial_hair = self.facial_hair_loss(x_facial_hair, facial_hair)
        loss_da_facial_hair = self.facial_hair_loss(x_da_facial_hair, facial_hair)
        
        loss_pose = self.pose_loss(x_pose, pose)
        loss_da_pose = self.pose_loss(x_da_pose, pose)
        
        loss_emotion = self.emotion_loss(x_emotion, emotion)
        loss_da_emotion = self.emotion_loss(x_da_emotion, emotion)
        
        loss_gender = self.gender_loss(x_gender, gender)
        loss_da_gender = self.gender_loss(x_da_gender, gender)
        
        loss_id = self.id_loss(x_id, id)
        
        total_loss =    loss_id + \
                        loss_gender * self.gender_weight + \
                        loss_da_gender * self.da_gender_weight + \
                        loss_emotion * self.emotion_weight + \
                        loss_da_emotion * self.da_emotion_weight + \
                        loss_pose * self.pose_weight + \
                        loss_da_pose * self.da_pose_weight + \
                        loss_facial_hair * self.facial_hair_weight + \
                        loss_da_facial_hair * self.da_facial_hair_weight + \
                        loss_occlusion * self.occlusion_weight + \
                        loss_da_occlusion * self.da_occlusion_weight + \
                        loss_spectacles * self.spectacles_weight + \
                        loss_da_spectacles * self.da_spectacles_weight
        
        return (
            total_loss,
            loss_id,
            loss_gender,
            loss_da_gender,
            loss_emotion,
            loss_da_emotion,
            loss_pose,
            loss_da_pose,
            loss_facial_hair,
            loss_da_facial_hair,
            loss_occlusion,
            loss_da_occlusion,
            loss_spectacles,
            loss_da_spectacles
        )
    