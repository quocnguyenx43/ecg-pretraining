import math
import sys
from typing import Iterable

import torch

from utils import misc as m
from utils.lr_schedule import adjust_learning_rate


def train_one_epoch_pretrain(model: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_scaler,
                    log_writer=None,
                    config=None):
    model.train()
    metric_logger = m.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', m.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = config['accum_iter']

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # # we use a per iteration (instead of per epoch) lr scheduler
        # if data_iter_step % accum_iter == 0:
        #     adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, config)

        input_ecg = samples['input_ecg'].type(torch.FloatTensor)
        lead_indices = samples['lead_indices'].type(torch.IntTensor)

        input_ecg = input_ecg.to(device, non_blocking=True)
        lead_indices = lead_indices.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            results = model(input_ecg, lead_indices)
        loss = results.get('loss')
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)
        loss = loss / accum_iter
        loss_scaler(loss,
                    optimizer,
                    parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]['lr']
        metric_logger.update(lr=lr)

        # if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
        #     """ We use epoch_1000x as the x-axis in tensorboard.
        #     This calibrates different curves when batch size changes.
        #     """
        #     epoch_1000x = int((epoch + data_iter_step / len(data_loader)) * 1000)
        #     log_writer.add_scalar('pretrain_loss', loss_value, epoch_1000x)
        #     log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    print('Averaged stats:', metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}