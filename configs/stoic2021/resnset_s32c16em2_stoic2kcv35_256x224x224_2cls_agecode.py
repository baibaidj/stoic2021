_base_ = [
    '../_base_/models/resnet_s32c16em2_stoic_256x224x224_2cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 16, workers_per_gpu= 16, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 3), 
            val=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 3), 
            test= dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv',
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 3, split = 'train'))

# pretrain_cp = 'work_dirs/simmim_convnext_s32c32em4_lung_224x224x192_100eps/latest.pth'

model = dict(
        backbone = dict(frozen_stages=4,
        #               init_cfg=dict(type='Pretrained', prefix='backbone.', 
        #               checkpoint=pretrain_cp, map_location = 'cpu')
                    ), 
        head = dict(
                    add_feat_dist = False, 
                    # is_multi_task = False, 
                    age_encoding = dict(type='SineAgeEncoding', 
                            temperature=64,
                            num_feats=256, normalize=True, max_age = 6),
                    verb = False,
                    # logit_dist_ratio = 0.3, 
                    loss=dict(type='FocalLossMultitask',
                        class_weight = (1.0, 1.5), 
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.5, 
                        loss_weight= 4.0),
                    ), 
        # test_cfg = dict(target_class = 0),                
    )

find_unused_parameters=True
load_from = 'work_dirs/resnset_s32c16em2_stoic2kcv35_256x224x224_2cls_agecode/best_auc_epoch_62.pth'
resume_from = None # 'work_dirs/resnset_s32c16em2_stoic2kcv35_256x224x224_2cls/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=2e-3, momentum=0.9, weight_decay=1e-3, 
                _delete_ = True, type='AdamW', lr=1e-4, weight_decay=1e-4
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                 policy='poly', power=0.99, min_lr=1e-5, 
                # policy='CosineAnnealing',  min_lr=1e-6, by_epoch=True, 
                warmup='linear', warmup_iters=100
                 )

runner = dict(type='EpochBasedRunner', max_epochs=64)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=2, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='auc', 
                save_best = 'auc', rule = 'greater'
                )

#
#  CUDA_VISIBLE_DEVICES=1 python tools/train.py configs/stoic2021/resnset_s32c16em2_stoic2kcv35_256x224x224_2cls_agecode.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/resnset_s32c16em2_stoic2kcv35_256x224x224_2cls_agecode.py 3

