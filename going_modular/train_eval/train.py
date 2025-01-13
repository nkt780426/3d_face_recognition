import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ..utils.roc_auc import compute_roc_auc
from ..utils.metrics import ProgressMeter
from ..utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ..utils.ModelCheckPoint import ModelCheckpoint
import os

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def fit(
    conf: dict,
    start_epoch: int,
    model: Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint
):
    log_dir = os.path.abspath(conf['checkpoint_dir'] + conf['type'] + '/logs')
    writer = SummaryWriter(log_dir=log_dir)
    device = conf['device']
    
    for epoch in range(start_epoch, conf['epochs']):
        
        train_loss = train_epoch(train_dataloader, model, criterion, optimizer, device)
        
        train_euclidean_auc, train_cosine_auc = compute_roc_auc(train_dataloader, model, device)
        test_euclidean_auc, test_cosine_auc = compute_roc_auc(test_dataloader, model, device)
        
        writer.add_scalar('Loss/train', train_loss, epoch+1)
        writer.add_scalars(main_tag='Cosine_auc', tag_scalar_dict={'train': train_cosine_auc, 'test': test_cosine_auc}, global_step=epoch+1)
        writer.add_scalars(main_tag='Euclidean_auc', tag_scalar_dict={'train': train_euclidean_auc, 'test': test_euclidean_auc}, global_step=epoch+1)

        train_metrics = [
            f"loss: {train_loss:.4f}", 
            f"auc_cos: {train_cosine_auc:.4f}",
            f"auc_eu: {train_euclidean_auc:.4f}",
        ]
        
        test_metrics = [
            f"auc_cos: {test_cosine_auc:.4f}",
            f"auc_eu: {test_euclidean_auc:.4f}",
        ]
        
        process = ProgressMeter(
            train_meters=train_metrics,
            test_meters=test_metrics,
            prefix=f"Epoch {epoch + 1}:"
        )
        
        process.display()
        
        model_checkpoint(model, optimizer, epoch + 1)
        early_stopping([test_cosine_auc, test_euclidean_auc], model, epoch + 1)
        
        # if early_max_stopping.early_stop and early_min_stopping.early_stop:
        #     break
        
        scheduler.step()
        
    writer.close()
    
def train_epoch(
    train_dataloader: DataLoader, 
    model: Module, 
    criterion: Module, 
    optimizer: Optimizer, 
    device: str,
    ):
    
    model.to(device)
    model.train()
    
    train_loss = 0
    
    for X, y in train_dataloader:
        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()

        logits = model(X)
            
        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()
            
        train_loss += loss.item()
            
    train_loss /=len(train_dataloader)
    
    return train_loss
