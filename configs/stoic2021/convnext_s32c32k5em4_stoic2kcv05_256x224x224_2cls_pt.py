_base_ = [
    '../_base_/models/convnext_s32c32k5em4_stoic_256x224x224.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 20, workers_per_gpu= 20, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0,), 
            val=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0,), 
            test= dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv',
                        img_dir=img_dir, prefix_dir = 'processed', cv_fold = 0,))

pretrain_cp = 'work_dirs/simmim_convnext_s32c32k5em4_stoic_224x192x192_100eps_exp3/latest.pth'

model = dict(
        backbone = dict(verb = False, 
                      init_cfg=dict(type='Pretrained', prefix='backbone.', 
                      checkpoint=pretrain_cp, map_location = 'cpu')
                    ), 
        head = dict(
                    add_feat_dist = False, 
                    # is_multi_task = False, 
                    # age_encoding = dict(type='SineAgeEncoding', 
                    #         temperature=32,
                    #         num_feats=256, normalize=True, max_age = 6),
                    verb = False,
                    loss=dict(type='FocalLossMultitask',
                        class_weight = (1.0, 1.5), 
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.5,  verb = False, 
                        loss_weight= 6.0),
                    ), 
        # test_cfg = dict(target_class = 0),    
    )

find_unused_parameters=True
load_from = None # 'work_dirs/convnext_s32c32k5em4_stoic2kcv05_256x224x224_2cls/latest.pth'
resume_from = None # 'work_dirs/convnext_s32c32k5em4_stoic2kcv05_256x224x224_2cls/latest.pth' 

# optimizer
optimizer = dict(
                type='SGD', lr=5e-3, momentum=0.9, weight_decay=1e-3, 
                # _delete_ = True, type='AdamW', lr=2e-4, weight_decay=1e-3
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                 policy='poly', power=0.99, min_lr=1e-5, 
                # policy='CosineAnnealing',  min_lr=1e-6, by_epoch=False, 
                warmup='linear', warmup_iters=100
                 )

runner = dict(type='EpochBasedRunner', max_epochs=200)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=2, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='severe_auc', 
                save_best = 'severe_auc', rule = 'greater'
                )

# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/stoic2021/convnext_s32c32k5em4_stoic2kcv05_256x224x224_2cls_pt.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/convnext_s32c32k5em4_stoic2kcv05_256x224x224_2cls_pt.py 3

