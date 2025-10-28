#!/bin/bash

export DETECTRON2_DATASETS=".."
ngpus=$(nvidia-smi --list-gpus | wc -l)

cfg_file=configs/nucls/maskformer2_R50_bs16_160k.yaml
base=results

base_lr=0.0001
iter=160000

soft_mask=False # mask softmax (True) or sigmoid (False)
soft_cls=False   # classifier softmax (True) or sigmoid( False)

num_prompts=0
deep_cls=True


comm_args="OUTPUT_DIR ${base}"

# Train base classes
# You can skip this process if you have a step0-checkpoint.
python train_net.py --num-gpus ${ngpus} --config-file ${cfg_file} ${comm_args} WANDB False

