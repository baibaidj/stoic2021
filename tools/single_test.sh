#!/usr/bin/env bash

cfg_na=$1
gpuix=$2
gpuix=${gpuix:-0}

model_dir=work_dirs/${cfg_na}
CUDA_VISIBLE_DEVICES=$gpuix python tools/test_med.py configs/stoic2021/${cfg_na}.py \
    $model_dir/latest.pth --eval auc --out $model_dir/result_by_pid_test.csv


# bash tools/single_test.sh convnext_s32c64_stoic2kcv05_192x192x160_2cls_embed4 4