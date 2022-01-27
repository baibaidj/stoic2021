_base_ = [
    '../_base_/models_med/convnext_s32c64_stoic_192x192x160_img.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 2, workers_per_gpu= 6, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', img_dir=img_dir, cv_fold = 0), 
            val=dict(sample_rate = 0.5, fn2imglist = 'stoic2021_case_info_split.csv', img_dir=img_dir, cv_fold = 0), 
            test= dict(sample_rate = 0.1, fn2imglist = 'stoic2021_case_info_split.csv',img_dir=img_dir, cv_fold = 0))
            
pretrain_cp = 'work_dirs/simmim_convnext_s32c64_lung_192x192x160_100eps_interp/latest.pth'

model = dict(
        backbone = dict(#backbone = dict(depths=[0, 4, 4, 4, 2]),
                      init_cfg=dict(type='Pretrained', prefix='backbone.', 
                      checkpoint=pretrain_cp, map_location = 'cpu')
                    ), 
        head = dict(use_sigmoid = True, 
                    # loss=dict(type='CrossEntropyLoss', loss_weight=2.0, 
                    #             class_weight = (0.3, 1.0, 1.5)
                    #         ),
                    loss_cls=dict(type='FocalLoss',
                        use_sigmoid=True,
                        gamma=2.0,
                        alpha=0.25, 
                        loss_weight=8.0),
                    ), 
        train_cfg= None, 
        test_cfg = None,                
    )

find_unused_parameters=True
load_from = None # 'work_dirs/convnext_s32c64_stoic2kcv05_192x192x160_2cls/latest.pth'
resume_from = None # 'work_dirs/convnext_s32c64_stoic2kcv05_192x192x160_2cls/latest.pth' 

# optimizer
optimizer = dict(
                # type='SGD', lr=1e-3, momentum=0.9, weight_decay=0.0001, 
                _delete_ = True, type='AdamW', lr=2e-4, weight_decay=0.02
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                #  policy='poly', power=0.99, min_lr=1e-5, 
                policy='CosineAnnealing',  min_lr=1e-5, by_epoch=True, 
                warmup='linear', warmup_iters=1000
                 )

runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=20, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='mAP', 
                save_best = 'mAP', rule = 'greater', 
                iou_thr=[0.2, 0.3])


# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/stoic2021/convnext_s32c64_stoic2kcv05_192x192x160_2cls.py 
# CUDA_VISIBLE_DEVICES=1,3,5 PORT=29135 bash ./tools/dist_train.sh configs/stoic2021/convnext_s32c64_stoic2kcv05_192x192x160_2cls.py 3 --gpus 3 #--no-validate

