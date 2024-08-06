"# ecg-pretraining" 

# env
conda remove -n venv -all
conda create -n venv python=x.x anaconda
conda activate venv
pip install pandas numpy tqdm einops torch scipy pyyam
pip3 install pipreqs
pip3 install pip-tools
pipreqs --savepath=requirements.txt && pip-compile


# commands
- converting data:
python convert_data_to_csv.py --input_datasets "g12c,chapman-shaoxing-ningbo" --output_dir_path "./data/index.csv"

- pretraining:
python main_pretrain.py --config_path "configs/pretrain.yaml"
tensorboard --logdir outputs/pretrain_1/log

- downstreaming: