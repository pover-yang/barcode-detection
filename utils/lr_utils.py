import math


def warmup_lr(max_epochs, warmup_epochs=None, warmup_factor=0.1):
    if not warmup_epochs:
        warmup_epochs = max_epochs // 20

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return warmup_factor + (1 - warmup_factor) * epoch / warmup_epochs
        else:
            return 1 / 2 * (1 + math.cos((epoch - warmup_epochs) / (max_epochs - warmup_epochs) * math.pi))

    return lr_lambda
