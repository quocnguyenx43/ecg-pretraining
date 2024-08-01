import torch
from torch import inf

import time
import datetime
from collections import defaultdict, deque


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values
    over a window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]
        
    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {:.4f} ({:.4f})".format(name, meter.median, meter.global_avg))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=''):
        # time each step
        data_time = SmoothedValue(fmt='{avg:.4f}') # each data load time
        iter_time = SmoothedValue(fmt='{avg:.4f}') # each iter run time

        # msg
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header, '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}', 'time: {time}', 'data: {data}',
        ]
        if torch.cuda.is_available():
            log_msg.append('mem: {memory:.0f}')
            log_msg.append('{meters}')
        else:
            log_msg.append('{meters}')
        log_msg = self.delimiter.join(log_msg)
        
        i = 0
        start_time = time.time()
        end_time = time.time()
        for obj in iterable:
            data_time.update(time.time() - start_time)
            yield obj
            iter_time.update(time.time() - start_time)

            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string, time=str(iter_time), data=str(data_time),
                        meters=str(self),
                        memory=torch.cuda.max_memory_allocated()/1024.0*1024.0)
                    )
                else:
                    print(log_msg.format(
                        i, len(iterable),
                        eta=eta_string, time=str(iter_time), data=str(data_time),
                        meters=str(self))
                    )

            end_time = time.time()
            i += 1

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        l = [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        total_norm = torch.norm(torch.stack(l), norm_type)
    return total_norm


def save_model(
        model, config, epoch, checkpoint_path,
        optimizer=None, loss_scaler=None, metrics=None,
    ):
    to_save = {
        'epoch': epoch,
        'model': model.state_dict(),
        'config': config,
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
        'metrics': metrics if metrics is not None else None,
    }
    torch.save(to_save, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_model(
        model, checkpoint_path,
        optimizer, loss_scaler
    ):
    # if checkpoint in hub
    if checkpoint_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location='cpu', check_hash=True
        )
    # if in local
    else:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    print('Model loaded!')

    if optimizer is not None and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Optimizer loaded!')
    
    if loss_scaler is not None and checkpoint['loss_scaler'] is not None:
        loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        print('Loss scaler loaded!')
    
    metrics = checkpoint.get('metrics', None)
    print(f"Checkpoint loaded from {checkpoint_path}")

    return epoch, model, optimizer, loss_scaler, metrics