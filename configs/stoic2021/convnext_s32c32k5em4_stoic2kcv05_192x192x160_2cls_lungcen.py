_base_ = [
    '../_base_/models/convnext_s32c32k5em4_stoic_192x192x160_2cls.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
    # '../_base_/swa.py',
]

img_dir = 'data/STOIC2021Round1'
data = dict(samples_per_gpu = 16, workers_per_gpu= 16, 
            train=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed_lung', cv_fold = 0,), 
            val=dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv', 
                        img_dir=img_dir, prefix_dir = 'processed_lung', cv_fold = 0,), 
            test= dict(sample_rate = 1.0, fn2imglist = 'stoic2021_case_info_split.csv',
                        img_dir=img_dir, prefix_dir = 'processed_lung', cv_fold = 0,))

# pretrain_cp = 'work_dirs/simmim_convnext_s32c32em4_lung_224x224x192_100eps/latest.pth'

model = dict(
        # backbone = dict(
        #               init_cfg=dict(type='Pretrained', prefix='backbone.', 
        #               checkpoint=pretrain_cp, map_location = 'cpu')
        #             ), 
        head = dict(
                    add_feat_dist = False, 
                    # is_multi_task = False, 
                #     age_encoding = dict(), 
                    verb = False,
                    loss=dict(type='CrossEntropyLoss', 
                                use_sigmoid = True, 
                                loss_weight=4.0, 
                                # class_weight = (0.8, 1.0), 
                                verbose = False,
                            ),
                    ), 
        # test_cfg = dict(target_class = 0),    
    )

find_unused_parameters=True
load_from = None # 'work_dirs/convnext_s32c32em4_stoic2kcv05_192x192x160_covid_enage_pretrain/latest.pth'
resume_from = None # 'work_dirs/convnext_s32c64_stoic2kcv05_192x192x160_2cls/latest.pth' 

# optimizer
optimizer = dict(
                type='SGD', lr=1e-2, momentum=0.9, weight_decay=1e-3, 
                # _delete_ = True, type='AdamW', lr=1e-3, weight_decay=1e-4
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
# fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
#             backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                 policy='poly', power=0.99, min_lr=1e-5, 
                # policy='CosineAnnealing',  min_lr=1e-6, by_epoch=True, 
                warmup='linear', warmup_iters=100
                 )

runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=1, max_keep_ckpts = 4)
# yapf:disable
log_config = dict(interval=2, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])

evaluation=dict(interval=2, start=0, metric='auc', 
                save_best = 'auc', rule = 'greater'
                )

# CUDA_VISIBLE_DEVICES=5 python tools/train.py configs/stoic2021/convnext_s32c32k5em4_stoic2kcv05_192x192x160_2cls_lungcen.py 
# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29024 bash ./tools/dist_train.sh configs/stoic2021/convnext_s32c32k5em4_stoic2kcv05_192x192x160_2cls_lungcen.py 3 --gpus 3

