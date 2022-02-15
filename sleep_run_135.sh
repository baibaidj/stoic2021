sleep 3h
#CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv25_256x224x224_2cls.py 3
CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv35_256x224x224_2cls.py 3
#CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv45_256x224x224_2cls.py 3

