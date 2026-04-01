# TRM-SSM

This repository contains the current training and evaluation code for TRM on event-based meshflow / optical flow tasks.

The main code paths in this repository are:

- HREM training: `train_TRM_HREM.py`
- HREM testing: `test_TRM_HREM.py`
- DSEC training: `train_TRM_DSEC.py`
- DSEC evaluation: `test_DSEC_split_eval.py`
- MVSEC training / evaluation utilities: `train_mvsec.py`, `test_mvsec.py`

## Environment

The code is written around Python 3.7 and PyTorch 1.10.1.

Recommended setup:

```bash
conda create -n trm python=3.7
conda activate trm
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

Notes:

- Install PyTorch separately first so you can match your CUDA version.
- Some optional components under `RVT/` are used by the CDC / SSM variants.

## Repository Structure

```text
config/                  training config files
loader/                  dataset loaders
model/                   TRM backbone and baseline implementations
RVT/                     temporal / adaptive SSM related modules
utils/                   logging, visualization, helper functions
train_TRM_HREM.py        HREM training entry
test_TRM_HREM.py         HREM test entry
train_TRM_DSEC.py        DSEC training entry
test_DSEC_split_eval.py  DSEC evaluation entry
train/                   shell scripts for running experiments
test/                    shell scripts for evaluation / ablation
```

## Datasets

### HREM

HREM is the main dataset used by the HREM training and testing scripts in this repository.

You need to prepare the HREM data locally and make sure the loader paths in your config match your environment.

### MVSEC

Download the HDF5 version of MVSEC:

- https://daniilidis-group.github.io/mvsec/download/

Event encoding utility:

```bash
python loader/MVSEC_encoder.py --only_event -dt=1
python loader/MVSEC_encoder.py --only_event -dt=4
python loader/MVSEC_encoder.py --only_event
```

### DSEC

The DSEC scripts expect the dataset under paths like:

```text
data/DSEC/train
data/DSEC/test_split.txt
```

You can override these paths with command-line arguments.

## HREM Training

Main entry:

```bash
python train_TRM_HREM.py --model_name TRM
```

Useful arguments in `train_TRM_HREM.py`:

- `--input_type dt1|dt4`
- `--batch_size`
- `--lr`
- `--wd`
- `--train_iters`
- `--val_iters`
- `--num_workers`
- `--output_dir`
- `--checkpoint_path`
- `--start-epoch`
- `--use_cdc`
- `--use_ssm`
- `--use_temporal_ssm`
- `--step_scale`
- `--ssm_state_dim`
- `--temporal_state_dim`
- `--blend_weight`

Example:

```bash
python train_TRM_HREM.py \
  --model_name TRM \
  --input_type dt4 \
  --batch_size 16 \
  --lr 2.5e-4 \
  --wd 5e-5 \
  --train_iters 50000 \
  --val_iters 2000 \
  --use_cdc \
  --use_ssm \
  --step_scale 2.0 \
  --ssm_state_dim 64
```

Quick shell entry:

```bash
bash train/train_trm.sh
```

Outputs are written under:

```text
exp_HREM_meshflow/
```

## HREM Testing

Main entry:

```bash
python test_TRM_HREM.py --model_name TRM --checkpoint_path /path/to/ckpt.pth.tar
```

Common arguments:

- `--input_type dt1|dt4`
- `--train_input_type`
- `--checkpoint_path`
- `--output_dir`
- `--visualize`
- `--vis_events`
- `--test_sequences`
- `--test_fps`
- `--use_cdc`
- `--use_ssm`
- `--use_temporal_ssm`
- `--step_scale`

Example:

```bash
python test_TRM_HREM.py \
  --model_name TRM \
  --input_type dt4 \
  --train_input_type dt4 \
  --checkpoint_path /path/to/best_ckpt.pth.tar \
  --use_cdc \
  --use_ssm \
  --step_scale 1.0 \
  --output_dir trm_eval
```

Provided shell script:

```bash
bash test/test_ssm_checkpoint.sh
```

Outputs are written under:

```text
test_output/
```

## DSEC Training

Main entry:

```bash
python train_TRM_DSEC.py --model_name TRM
```

Common arguments:

- `--dataset_root`
- `--dataset_root_val`
- `--train_split_file`
- `--val_split_file`
- `--num_bins`
- `--batch_size`
- `--train_iters`
- `--val_iters`
- `--flow_scale`
- `--step_scale`
- `--use_cdc`
- `--use_ssm`
- `--use_temporal_ssm`

Example:

```bash
python train_TRM_DSEC.py \
  --model_name TRM \
  --dataset_root data/DSEC/train \
  --dataset_root_val data/DSEC/train \
  --num_bins 5 \
  --batch_size 32 \
  --train_iters 50000 \
  --val_iters 2000 \
  --use_cdc \
  --use_ssm \
  --step_scale 1.0
```

## DSEC Evaluation

Main entry:

```bash
python test_DSEC_split_eval.py --ckpt_path /path/to/lasted_ckpt.pth.tar
```

Example:

```bash
python test_DSEC_split_eval.py \
  --ckpt_path exp_DSEC_meshflow/your_run/lasted_ckpt.pth.tar \
  --config_path exp_DSEC_meshflow/your_run/config.json \
  --train_cfg_path exp_DSEC_meshflow/your_run/train_config.json \
  --train_root data/DSEC/train \
  --split_file data/DSEC/test_split.txt
```

Shell entry:

```bash
bash test/test_dsec_split_eval.sh
```

## MVSEC

The repository still keeps the original MVSEC training / evaluation utilities:

```bash
python train_mvsec.py
python test_mvsec.py
```

## Pretrained Weights

This repository should only store code.

Do not commit:

- dataset files
- checkpoints
- `*.pth`
- `*.pth.tar`
- `*.tar`
- `*.zip`
- `__pycache__/`
- `.ipynb_checkpoints/`
- training logs and generated outputs

Place checkpoints locally and pass them through `--checkpoint_path`.

