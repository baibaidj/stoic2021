
export PYTHONPATH=':'
gpuix=$1 #${PORT:-29500}
numfold=$2



# model_name=resnset_s32c16em2_stoic2kcv05_256x224x224_2cls_agecode
# best_weight=best_auc_epoch_6

cvfold=4
model_name=resnset_s32c16em2_stoic2kcv${cvfold}5_256x224x224_2cls_agecode_ft
# best_weight=best_severe_auc_epoch_2

# model_name=resnset_s32c16em2_stoic2kcv15_256x224x224_2cls_agecode_ft
# best_weight=best_severe_auc_epoch_10

# model_name=resnset_s32c16em2_stoic2kcv25_256x224x224_2cls_agecode_ft
# best_weight=best_severe_auc_epoch_18

# model_name=resnset_s32c16em2_stoic2kcv35_256x224x224_2cls_agecode_ft
# best_weight=best_severe_auc_epoch_26

# model_name=resnset_s32c16em2_stoic2kcv45_256x224x224_2cls_agecode_ft
best_weight=best_severe_auc_epoch_6



work_dirs=work_dirs 
data_rt=data/STOIC2021Round1
prefix_dir=processed
split=test

gpuix=${gpuix:-0}
numfold=${numfold:-3}
foldix=${FOLDIX:-0}
for (( foldix = 0; foldix < $numfold; foldix++ )) # $numfold
do {
    sleeptime=$(( 1*foldix + 0 ))
    gpuix_exe=$gpuix #$(( gpuix + $foldix))
    echo GPU$gpuix_exe-Foldix$foldix/$numfold-WaitStart$sleeptime
    if [[ $foldix -gt 0 ]]; then # run in parralel but not start together
        sleep $sleeptime
    fi
    CUDA_VISIBLE_DEVICES=$gpuix_exe python scripts/infer_clstask.py \
    --repo-rt $work_dirs --model-name $model_name --best-weight $best_weight \
    --data-rt $data_rt --prefix-dir $prefix_dir --split $split --cv-fold ${cvfold} \
    --gpu-ix 0 --fold-ix $foldix --num-fold $numfold --verbose
    } &
done
wait

# CUDA_VISIBLE_DEVICES=$gpuix python scripts/infer_clstask.py \
#     --repo-rt $work_dirs --model-name $model_name --best-weight $best_weight \
#     --data-rt $data_rt --prefix-dir $prefix_dir --split $split --cv-fold ${cvfold} \
#     --gpu-ix 0 --fold-ix $foldix --num-fold $numfold --verbose

# bash scripts/infer_stoic.sh 3 6

# python tools/model_converters/publish_model.py $work_dirs/$model_name/${weightfile} $work_dirs/$model_name/publish_22.pth