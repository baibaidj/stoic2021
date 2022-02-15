_base_ = [
    '../_base_/models/convnext_s16c32k5em4_stoic_256x224x224.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 16, workers_per_gpu= 16, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0, target_class = (0,)), 
            val=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0, target_class = (0,)), 
            test= dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv',
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0, target_class = (0,)))

# pretrain_cp = 'work_dirs/simmim_convnext_s32c32em4_lung_224x224x192_100eps/latest.pth'

model = dict(
        backbone = dict(verb = False
                    #   init_cfg=dict(type='Pretrained', prefix='backbone.', 
                    #   checkpoint=pretrain_cp, map_location = 'cpu')
                    ), 
        head = dict(
                    add_feat_dist = False, 
                    num_classes = 1, 
                    # is_multi_task = False, 
                    # age_encoding = dict(type='SineAgeEncoding', 
                    #         temperature=32,
                    #         num_feats=256, normalize=True, max_age = 6),
                    verb = False,
                    loss=dict(type='CrossEntropyLoss', 
                                use_sigmoid = False, 
                                loss_weight=4.0, 
                                class_weight = (0.8, 1.0), 
                                verbose = False,
                            ),
                    ), 
        test_cfg = dict(target_class = 0),    
    )

find_unused_parameters=True
load_from = 'work_dirs/convnext_s16c32k5em4_stoic2kcv05_256x224x224_2cls_lungcen/latest.pth'
resume_from = None # 'work_dirs/convnext_s16c32k5em4_stoic2kcv05_256x224x224_2cls_lungcen/latest.pth' 

# optimizer
optimizer = dict(
                type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-3, 
                # _delete_ = True, type='AdamW', lr=1e-3, weight_decay=1e-4
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

runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=2, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='auc', 
                save_best = 'auc', rule = 'greater'
                )

# CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/stoic2021/convnext_s16c32k5em4_stoic2kcv05_256x224x224_1cls_lungcen.py 
# CUDA_VISIBLE_DEVICES=2,4 PORT=29024 bash ./tools/dist_train.sh configs/stoic2021/convnext_s16c32k5em4_stoic2kcv05_256x224x224_1cls_lungcen.py 2

