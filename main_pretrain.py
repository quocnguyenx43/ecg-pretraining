import os
import numpy as np
import yaml
import time
import argparse
import json

import torch
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

from utils import dataset as d
from utils import optimizer as o
from utils import functions as f
from utils import misc as m

from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from models.model import model_test
from utils.engine import train_one_epoch_pretrain



def parse() -> dict:
    parser = argparse.ArgumentParser('ECG pre-training args')
    parser.add_argument('--config_path', default='./configs/pretrain.yaml', type=str, help='YAML config file path for pretraining')

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in vars(args).items():
        if v:
            config[k] = v
    return config


def main(config) -> None:
    # Configs
    print(yaml.dump(config, default_flow_style=False, sort_keys=False))
    device = torch.device(config['device'])
    f.setup_seed(config['seed'])

    # if config['output_dir']:
    #     output_dir = config['output_dir'] + config['exp_name'] + '/'
    #     log_dir = output_dir + 'log/'
    #     model_dir = output_dir + 'models/'

    #     os.makedirs(output_dir, exist_ok=True)
    #     os.makedirs(log_dir, exist_ok=True)
    #     os.makedirs(model_dir, exist_ok=True)

    #     log_writer = SummaryWriter(log_dir=log_dir)
    # else:
    #     print('There not log_writer!')
    #     output_dir = None
    #     log_dir = None
    #     model_dir = None
    #     log_writer = None

    # Data preparation
    train_dataset = d.build_dataset(config['dataset'], split='train')
    train_dataloader = d.build_dataloader(train_dataset, mode='train', **config['dataloader'])

    # Model
    model = model_test(**config['model'])
    model.to(device)
    optimizer = o.get_optimizer(config['train'], model)
    # loss_scaler = NativeScaler()
        
    # # Load previous checkpoint
    # total_epochs = config['train']['total_epochs']
    # last_checkpoint_path = config['resume']
    # if last_checkpoint_path:
    #     last_epoch, model, optimizer, loss_scaler, metrics = m.load_model(
    #         model, last_checkpoint_path,
    #         optimizer, loss_scaler
    #     )
    #     model.to(device)
    #     start_epoch = last_epoch + 1
    #     end_epoch = start_epoch + total_epochs
    # else:
    #     start_epoch = 0
    #     end_epoch = total_epochs

    # loss_scaler = loss_scaler
    
    # # Training
    # print(f"Start training for {total_epochs} epochs")
    # if log_writer is not None:
    #     print('log_dir: {}'.format(log_writer.log_dir))
    # print()

    for epoch in range(10):
        # train 1 epoch
        # train_stats = train_one_epoch_pretrain(
        #     model, train_dataloader,
        #     epoch, end_epoch, device,
        #     optimizer, loss_scaler,
        #     log_writer, config['train']
        # )

        for batch in train_dataloader:

            input_ecg = batch['input_ecg'].type(torch.FloatTensor).to(device, non_blocking=True)
            lead_indices = batch['lead_indices'].type(torch.IntTensor).to(device, non_blocking=True)
            
            optimizer.zero_grad()
            results = model(input_ecg, lead_indices)

            loss = results['loss']
            loss.backward()
            optimizer.step()
        print(1)

        # # write log as txt file
        # log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}
        # if output_dir:
        #     if log_writer is not None:
        #         log_writer.flush()
        #     with open(os.path.join(log_dir, 'log.txt'), 'a', encoding="utf-8") as file:
        #         file.write(json.dumps(log_stats) + '\n')

        # # save model
        # if output_dir and (epoch % 20 == 0 or epoch == end_epoch - 1):
        #     checkpoint_path = model_dir + f'checkpoint-{epoch}.pth'
        #     m.save_model(
        #         model, config, epoch, checkpoint_path,
        #         optimizer, loss_scaler#, metrics,
        #     )
        
        # print()
        

if __name__ == '__main__':
    config = parse()
    main(config)

# python main_pretrain.py --config_path "configs/pretrain.yaml"
# tensorboard --logdir outputs/pretrain_1/log