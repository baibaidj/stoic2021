#!/usr/bin/env bash

cfg_na=$1
gpuix=$2
gpuix=${gpuix:-0}

model_dir=work_dirs/${cfg_na}
CUDA_VISIBLE_DEVICES=$gpuix python tools/test_med.py configs/stoic2021/${cfg_na}.py \
    $model_dir/latest.pth --eval auc --out $model_dir/result_cv05_train.csv


# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv05_192x192x144_2cls_agecode 4
# bash tools/single_test.sh convnext_s32c32em4_stoic2kcv05_224x224x192_covid_scratch 4
# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_ft 4
# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv15_256x224x224_2cls_agecode 5
# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv25_256x224x224_2cls_agecode 5
# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv35_256x224x224_2cls_agecode 5
# bash tools/single_test.sh resnset_s32c16em2_stoic2kcv45_256x224x224_2cls_agecode 4
# resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode_ft