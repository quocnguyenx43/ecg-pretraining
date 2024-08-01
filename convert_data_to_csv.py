import os
import re
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd

from utils import functions as f


# _DATASETS = {
#     # pretraining datasets
#     'g12c': '../../input/georgia-12lead-ecg-challenge-database/Georgia',
#     'chapman-shaoxing-ningbo': '../../input/shaoxing-and-ningbo-first-hospital-database/WFDB_ShaoxingUniv',
#     # downstreaming datasets
#     'ptbxl': '../../input/ptbxl-electrocardiography-database/WFDB',
#     'cpsc2018': '../../input/china-physiological-signal-challenge-in-2018/Training_WFDB',
# }

_DATASETS = {
    # pretraining datasets
    'g12c': './data/georgia-12lead-ecg-challenge-database/Georgia',
    'chapman-shaoxing-ningbo': './data/shaoxing-and-ningbo-first-hospital-database/WFDB_ShaoxingUniv',
    # downstreaming datasets
    'ptbxl': './data/ptbxl-electrocardiography-database/WFDB',
    'cpsc2018': './data/china-physiological-signal-challenge-in-2018/Training_WFDB',
}

def get_parser():
    parser = argparse.ArgumentParser(description='Process to data to CSV file.')
    parser.add_argument('-i', '--input_datasets', type=str, required=True, default='g12c,chapman-shaoxing-ningbo')
    parser.add_argument('--output_dir_path', type=str, default='./data/index.csv')
    args = parser.parse_args()
    args = vars(args)
    args['input_datasets'] = args['input_datasets'].split(',')
    return args

def process(args):
    sample_paths = []
    for dir in args['input_datasets']:
        dir = _DATASETS[dir]
        s = set([dir + '/' + file.split('.')[0] for file in os.listdir(dir)])
        sample_paths += s
    print(f"Found {len(sample_paths)} records.")

    index_df = pd.DataFrame(columns=['path', 'fs', 'length', 'label'])
    num_sample_saved = 0
    for sample_path in tqdm(sample_paths):
        _, data_header = f.load_data(sample_path)

        splits = data_header[0].split(' ')
        fs = int(splits[2])
        length = int(splits[3])
        label = re.search(r'#Dx:\s*([0-9,]+)', ' '.join(data_header)).group(1)

        index_df.loc[num_sample_saved] = [sample_path, fs, length, label]
        num_sample_saved += 1

    print(f"Saved.")
    index_df.to_csv(args['output_dir_path'], index=False)

if __name__ == "__main__":
    print('Processing ...')
    process(get_parser())
    print('Done!')

# python convert_data_to_csv.py --input_datasets "g12c,chapman-shaoxing-ningbo" --output_dir_path "./data/index.csv"