class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.total_samples = 0
        self.avg = 0

    # val: giá trị accuracy, loss trong 1 batch train, n là batch_size
    def update(self, val, batch_size):
        self.val += val
        self.total_samples += batch_size

    def compute(self):
        self.avg = self.val / self.total_samples
    
    # format kết quả train
    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)

# Hiển thị các thông tin trong quá trình training. Kết hợp với tham số AverageMeter ở trên.
class ProgressMeter(object):
    def __init__(self, train_meters, test_meters, prefix=""):
        self.train_meters = train_meters
        self.test_meters = test_meters
        self.prefix = prefix

    def display(self):
        print(f"{self.prefix}")
        
        train_metrics = " | ".join(str(meter) for meter in self.train_meters)
        print(f"\ttrain: {train_metrics}")

        val_metrics = " | ".join(str(meter) for meter in self.test_meters)
        print(f"\ttest: {val_metrics}")