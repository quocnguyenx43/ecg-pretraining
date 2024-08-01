import torch
from torch.optim import Optimizer, SGD, AdamW

def get_optimizer(config: dict, model: torch.nn.Module) -> Optimizer:
    opt_name = config['optimizer']
    lr = config['lr']
    weight_decay = config['weight_decay']
    kwargs = config.get('optimizer_kwargs', {})
    if opt_name == "sgd":
        momentum = kwargs.get('momentum', 0)
        return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == "adamw":
        betas = kwargs.get('betas', (0.9, 0.999))
        if isinstance(betas, list):
            betas = tuple(betas)
        eps = kwargs.get('eps', 1e-8)
        return AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
