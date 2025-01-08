import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from ...utils.roc_auc import compute_roc_auc
from ...utils.metrics import ProgressMeter
from ...utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ...utils.ModelCheckPoint import ModelCheckpoint
import os

def fit(
    conf:dict,
    start_epoch: int,
    model: Module, 
    triplet_train_loader: DataLoader, 
    triplet_test_loader: DataLoader, 
    criterion: Module, 
    optimizer: Optimizer, 
    scheduler, 
    epochs: int, 
    device:str, 
    roc_train_loader: DataLoader, 
    roc_test_loader: DataLoader,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint,
):
    log_dir = os.path.abspath(conf['checkpoint_dir']+ conf['type'] + '/logs')
    writer = SummaryWriter(log_dir=log_dir)
    model.to(device)
    
    for epoch in range(start_epoch, epochs):
        scheduler.step()

        # Train stage
        train_loss = train_epoch(triplet_train_loader, model, criterion, optimizer, device)
        test_loss = test_epoch(triplet_test_loader, model, criterion, device)
        
        train_euclidean_auc, train_cosine_auc = compute_roc_auc(roc_train_loader, model, device)
        test_euclidean_auc, test_cosine_auc = compute_roc_auc(roc_test_loader, model, device)
    
        writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train': train_loss, 'test': test_loss}, global_step=epoch+1)
        writer.add_scalars(main_tag='Cosine_auc', tag_scalar_dict={'train': train_cosine_auc, 'test': test_cosine_auc}, global_step=epoch+1)
        writer.add_scalars(main_tag='Euclidean_auc', tag_scalar_dict={'train': train_euclidean_auc, 'test': test_euclidean_auc}, global_step=epoch+1)

        train_metrics = [
            f"loss: {train_loss:.4f}", 
            f"auc_cos: {train_cosine_auc:.4f}",
            f"auc_eu: {train_euclidean_auc:.4f}",
        ]
        
        test_metrics = [
            f"loss: {test_loss:.4f}", 
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

    writer.close()


def train_epoch(
    triplet_train_loader: DataLoader, 
    model: Module, 
    criterion: Module, 
    optimizer: Optimizer, 
    device: str
):

    model.train()
    losses = []
    train_loss = 0
    for i, X in enumerate(triplet_train_loader):
        try:
            X = X.to(device)
            optimizer.zero_grad()

            anchors, positives, negatives = model(X)

            loss = criterion(anchors, positives, negatives)

            loss.backward()

            optimizer.step()
            losses.append(loss.item())
            train_loss += loss.item()
        except Exception as e:
            print(f"Train Error in batch {i+1}: {e}")
            break

    train_loss /= len(triplet_train_loader)

    return train_loss


def test_epoch(
    triplet_test_loader: DataLoader, 
    model: Module, 
    criterion: Module, 
    device: str
):

    with torch.no_grad():
        model.eval()
        test_loss = 0
        
        for i, X in enumerate(triplet_test_loader):
            try: 
            
                X = X.to(device)
                
                anchors, positives, negatives = model(X)

                loss = criterion(anchors, positives, negatives)
                
                test_loss += loss.item()
                
            except Exception as e:
                print(f"Test rrror in batch {i+1}: {e}")
                break

        test_loss /= len(triplet_test_loader)
        
    return test_loss
