import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
import time

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def compute_roc_auc(
    dataloader: torch.utils.data.DataLoader, 
    model: torch.nn.Module, 
    device: str
):
    
    model.eval()
    with torch.no_grad():
        embeddings_list = []
        for batch in dataloader:
            images, ids = batch
            images = images.to(device)
            embeddings = model.get_embedding(images)
            
            embeddings_list.append((ids, embeddings))
        
        # Concatenate all embeddings into one tensor
        all_ids = torch.cat([x[0] for x in embeddings_list], dim=0)
        all_embeddings = torch.cat([x[1] for x in embeddings_list], dim=0)
        
        euclidean_scores = []
        euclidean_labels = []
        cosine_scores = []
        cosine_labels = []

        # Compute pairwise Euclidean distance and cosine similarity
        all_embeddings_norm = all_embeddings / all_embeddings.norm(p=2, dim=1, keepdim=True)
        euclidean_distances = torch.cdist(all_embeddings, all_embeddings, p=2)  # Euclidean distance matrix
        cosine_similarities = torch.mm(all_embeddings_norm, all_embeddings_norm.t())  # Cosine similarity matrix
        
        # Compute labels (same id = 0, different id = 1)
        labels = (all_ids.unsqueeze(1) == all_ids.unsqueeze(0)).int().to(device)

        # Flatten and filter results
        euclidean_scores = euclidean_distances[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        euclidean_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        
        cosine_scores = cosine_similarities[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        cosine_labels = labels[torch.triu(torch.ones_like(labels), diagonal=1) == 1].cpu().numpy()
        
        # Compute ROC AUC for Euclidean distance
        euclidean_true_labels = 1 - np.array(euclidean_labels)
        euclidean_pred_scores = np.array(euclidean_scores)
        fpr_euclidean, tpr_euclidean, _ = roc_curve(euclidean_true_labels, euclidean_pred_scores)
        roc_auc_euclidean = auc(fpr_euclidean, tpr_euclidean)

        # Compute ROC AUC for Cosine similarity
        cosine_true_labels = np.array(cosine_labels)
        cosine_pred_scores = np.array(cosine_scores)
        fpr_cosine, tpr_cosine, _ = roc_curve(cosine_true_labels, cosine_pred_scores)
        roc_auc_cosine = auc(fpr_cosine, tpr_cosine)
        
        # # Calculate accuracy for Euclidean distance
        # euclidean_optimal_idx = np.argmax(tpr_euclidean - fpr_euclidean) # Chọn ngưỡng tại điểm có giá trị tpr - fpr lớn nhất trên đường ROC, vì đây là nơi tối ưu hóa sự cân bằng giữa tỷ lệ phát hiện (TPR) và tỷ lệ báo động giả (FPR).
        # euclidean_optimal_threshold = thresholds_euclidean[euclidean_optimal_idx]
        # euclidean_pred_labels = (euclidean_pred_scores >= euclidean_optimal_threshold).astype(int)
        # euclidean_accuracy = accuracy_score(euclidean_true_labels, euclidean_pred_labels)

        # # Calculate accuracy for Cosine similarity
        # cosine_optimal_idx = np.argmax(tpr_cosine - fpr_cosine)
        # cosine_optimal_threshold = thresholds_cosine[cosine_optimal_idx]
        # cosine_pred_labels = (cosine_pred_scores >= cosine_optimal_threshold).astype(int)
        # cosine_accuracy = accuracy_score(cosine_true_labels, cosine_pred_labels)
        
    return roc_auc_euclidean, roc_auc_cosine