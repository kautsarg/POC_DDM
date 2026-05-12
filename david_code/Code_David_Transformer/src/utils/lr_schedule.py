"""
Learning rate scheduling utilities.
"""

import math


def inv_lr_scheduler(optimizer, iter_num, gamma, power, lr=0.001, weight_decay=0.0005):
    """
    Inverse learning rate scheduler.

    lr = lr * (1 + gamma * iter_num) ** (-power)

    Args:
        optimizer: PyTorch optimizer
        iter_num: Current iteration number
        gamma: Scaling factor
        power: Power factor
        lr: Base learning rate
        weight_decay: Weight decay factor

    Returns:
        Updated optimizer
    """
    lr = lr * (1 + gamma * iter_num) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group.get('lr_mult', 1.0)
        param_group['weight_decay'] = weight_decay * param_group.get('decay_mult', 1.0)
    return optimizer


def cosine_warmup_lr_scheduler(optimizer, iter_num, lr, T_max,
                                warmup_iters=500, lr_min=1e-6, weight_decay=0.0005):
    """
    Cosine annealing with linear warmup.

    0 .. warmup_iters : LR rises linearly from 0 to lr
    warmup_iters .. T_max : LR follows cosine curve from lr down to lr_min
    """
    if iter_num < warmup_iters:
        current_lr = lr * (iter_num + 1) / warmup_iters
    else:
        progress = (iter_num - warmup_iters) / max(1, T_max - warmup_iters)
        current_lr = lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr * param_group.get('lr_mult', 1.0)
        param_group['weight_decay'] = weight_decay * param_group.get('decay_mult', 1.0)
    return optimizer


schedule_dict = {
    "inv": inv_lr_scheduler,
    "cosine_warmup": cosine_warmup_lr_scheduler,
}

