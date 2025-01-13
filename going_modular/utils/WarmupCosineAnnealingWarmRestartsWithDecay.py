from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingWarmRestarts, LambdaLR

class WarmupCosineAnnealingWarmRestartsWithDecay(_LRScheduler):
    def __init__(self, optimizer, warmup_iters, T_0, T_mult=2, eta_min=0, decay_factor=0.5):
        self.optimizer = optimizer
        self.warmup_iters = warmup_iters
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.decay_factor = decay_factor
        self.min_delta = 1e-6

        # Store the original base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.current_base_lrs = self.base_lrs.copy()

        # Define warmup scheduler
        self.warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: epoch / warmup_iters if epoch < warmup_iters else 1.0
        )

        # Cosine Annealing scheduler
        self.cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=self.T_0, T_mult=self.T_mult, eta_min=self.eta_min)

    def step(self, epoch):
        if epoch < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            # Update learning rates after each restart
            restart_epoch = epoch - self.warmup_iters
            if restart_epoch == 0 or restart_epoch % self.cosine_scheduler.T_0 == 0:
                self.current_base_lrs = [
                    max(lr * self.decay_factor, self.eta_min + self.min_delta) for lr in self.current_base_lrs
                ]
                for i, param_group in enumerate(self.optimizer.param_groups):
                    param_group['lr'] = self.current_base_lrs[i]

            self.cosine_scheduler.step(epoch - self.warmup_iters)

    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
