import math
import sys
from typing import Iterable

import torch

from utils import misc as m
from utils.lr_schedule import adjust_learning_rate


def train_one_epoch_pretrain(
        model: torch.nn.Module, data_loader: Iterable,
        epoch: int, end_epoch: int, device: torch.device,
        optimizer: torch.optim.Optimizer, loss_scaler=None, 
        log_writer=None, config=None,
    ):
    
    model.train()
    optimizer.zero_grad()

    metric_logger = m.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', m.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, print_header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if step % accum_iter == 0:
            adjust_learning_rate(optimizer, step / len(data_loader) + epoch, end_epoch, config)

        input_ecg = batch['input_ecg']
        lead_indices = batch['lead_indices']
        
        input_ecg = input_ecg.to(device, non_blocking=True).type(torch.FloatTensor)
        lead_indices = lead_indices.to(device, non_blocking=True).type(torch.IntTensor)
        print(input_ecg)

        with torch.cuda.amp.autocast():
            results = model(input_ecg, lead_indices)
            
        loss = results.get('loss').to(device, non_blocking=True)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss = loss / accum_iter

        is_updating_model = (step + 1) % accum_iter == 0
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=is_updating_model)
        if is_updating_model:
            optimizer.zero_grad()

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=lr)
        
        if log_writer is not None and is_updating_model:
            # 1000 is the x-axis
            epoch_1000x = int((epoch + step / len(data_loader)) * 1000)
            log_writer.add_scalar('pretrain_loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
