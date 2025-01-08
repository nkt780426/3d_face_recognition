import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from .train_id_acc import train_id_accuracy
from ...utils.roc_auc import compute_roc_auc
from ...utils.metrics import AverageMeter, ProgressMeter
from ...utils.MultiMetricEarlyStopping import MultiMetricEarlyStopping
from ...utils.ModelCheckPoint import ModelCheckpoint
import os

# Đặt seed toàn cục
seed = 42
torch.manual_seed(seed)

def fit(
    conf,
    start_epoch: int,
    model: Module,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    scheduler,
    early_stopping: MultiMetricEarlyStopping,
    model_checkpoint: ModelCheckpoint
):
    log_dir = os.path.abspath('checkpoint/magface/'+ conf['type'] + '/logs')
        
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, conf['epochs']):
        train_loss = AverageMeter('loss', ':.3f')
        train_loss_id = AverageMeter('loss id', ':6.2f')
        train_acc_top1 = AverageMeter('top_1_acc', ':6.4f')
        train_acc_top5 = AverageMeter('top_5_acc', ':6.4f')
        
        model.train()
        
        for input, target in train_dataloader:
            input = input.to(device)
            target = target.to(device)
            logits, x_norm = model(input)
            
            # caculate metric
            loss_id, loss_g = criterion(logits, target, x_norm)
            loss = loss_id + loss_g
            acc1, acc5 = train_id_accuracy(logits[0], target)
            
            # update metric
            batch_size = input.size(0)
            train_loss.update(loss.item(), batch_size)
            train_loss_id.update(loss_id.item(), batch_size)
            train_acc_top1.update(acc1, batch_size)
            train_acc_top5.update(acc5, batch_size)
            
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_loss.compute()
        train_loss_id.compute()
        train_acc_top1.compute()
        train_acc_top5.compute()
        
        train_euclidean_accuracy, train_cosine_accuracy, train_auc_euclidean, train_auc_cosine = compute_roc_auc(train_dataloader, model, device)
        
        test_euclidean_accuracy, test_cosine_accuracy, test_auc_euclidean, test_auc_cosine = compute_roc_auc(val_dataloader, model, device)
        
        # Ghi các giá trị vào TensorBoard
        writer.add_scalar("Loss/train", train_loss.avg, epoch + 1)
        writer.add_scalar("Loss/train_id", train_loss_id.avg, epoch + 1)
        writer.add_scalars(main_tag="Train_accuracy_id", tag_scalar_dict = {'top_1': train_acc_top1.avg, 'top_5': train_acc_top5.avg}, global_step=epoch + 1)
        writer.add_scalars(main_tag='Cosine_AUC', tag_scalar_dict = {'train': train_auc_cosine, 'val': test_auc_cosine}, global_step=epoch + 1)
        writer.add_scalars(main_tag="Cosine_ACC", tag_scalar_dict = {'train': train_cosine_accuracy, 'val': test_cosine_accuracy}, global_step=epoch + 1)
        writer.add_scalars(main_tag='Euclidean_AUC', tag_scalar_dict = {'train': train_auc_euclidean, 'val': test_auc_euclidean}, global_step=epoch + 1)
        writer.add_scalars(main_tag="Euclidean_ACC", tag_scalar_dict = {'train': train_euclidean_accuracy, 'val': test_euclidean_accuracy}, global_step=epoch + 1)
        
        train_metrics = [
            train_loss, 
            train_loss_id, 
            train_acc_top1,
            train_acc_top5,
            f"cos_auc: {train_auc_cosine:.4f}",
            f"cos_acc: {train_cosine_accuracy:.4f}",
            f"eu_auc: {train_auc_euclidean:.4f}",
            f"eu_acc: {train_euclidean_accuracy:.4f}",
        ]
        
        test_metrics = [
            f"cos_auc: {test_auc_cosine:.4f}",
            f"cos_acc: {test_cosine_accuracy:.4f}",
            f"eu_auc: {test_auc_euclidean:.4f}",
            f"eu_acc: {test_euclidean_accuracy:.4f}",
        ]
        
        process = ProgressMeter(
            train_meters=train_metrics,
            test_meters=test_metrics,
            prefix=f"Epoch {epoch + 1}:"
        )
        
        process.display()
        
        model_checkpoint(model, optimizer, epoch + 1)
        scheduler.step(epoch+1)
        early_stopping([test_auc_cosine, test_cosine_accuracy, test_auc_euclidean, test_euclidean_accuracy], model, epoch+1)
        if early_stopping.early_stop:
            break
        
    writer.close()