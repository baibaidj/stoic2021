_base_ = [
    '../_base_/models/convnext_s32c64_stoic_192x192x160_img_embed4.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 12, workers_per_gpu= 18, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', img_dir=img_dir, cv_fold = 0), 
            val=dict(sample_rate = 0.5, fn2imglist = 'stoic2021_case_info_split.csv', img_dir=img_dir, cv_fold = 0), 
            test= dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv',img_dir=img_dir, cv_fold = 0))

# pretrain_cp = 'work_dirs/simmim_convnext_s32c64_lung_192x192x160_100eps_interp/latest.pth'

model = dict(
        # backbone = dict(#backbone = dict(depths=[0, 4, 4, 4, 2]),
        #               init_cfg=dict(type='Pretrained', prefix='backbone.', 
        #               checkpoint=pretrain_cp, map_location = 'cpu')
        #             ), 
        head = dict(
                    add_feat_dist = False, 
                    loss=dict(type='FocalLossMultitask',
                        class_weight = (1.0, 1.25), 
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.6, 
                        loss_weight= 2.0),
                    # verb = True,
                    # loss=dict(type='CrossEntropyLoss', loss_weight=2.0, 
                    #             class_weight = (0.3, 1.0, 1.5)
                    #         ),
                    ), 
        train_cfg= None, 
        test_cfg = None,                
    )

find_unused_parameters=True
load_from = 'work_dirs/convnext_s32c64_stoic2kcv05_192x192x160_2cls_embed4/latest.pth'
resume_from = None # 'work_dirs/convnext_s32c64_stoic2kcv05_192x192x160_2cls/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=2e-4, weight_decay=1e-3
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                #  policy='poly', power=0.99, min_lr=1e-5, 
                policy='CosineAnnealing',  min_lr=1e-6, by_epoch=True, 
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

# CUDA_VISIBLE_DEVICES=4 python tools/train.py configs/stoic2021/convnext_s32c64_stoic2kcv05_192x192x160_2cls_nodist.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/convnext_s32c64_stoic2kcv05_192x192x160_2cls_nodist.py 3 --gpus 3

