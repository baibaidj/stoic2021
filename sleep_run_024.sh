sleep 6h
CUDA_VISIBLE_DEVICES=0,2,4 PORT=29024 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv45_256x224x224_2cls.py 3
