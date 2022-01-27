_base_ = [
    '../_base_/models_med/simmim_convnext_s32c64_lung_192x192x160.py', # s32= stride32
    # '../_base_/datasets/ribfrac_instance_semantic.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

data = dict(samples_per_gpu = 6, workers_per_gpu= 9, 
            train=dict(sample_rate = 1.0, json_filename = 'case_info_5sources.csv', split='train'), 
            val=dict(sample_rate = 1.0, json_filename = 'case_info_5sources.csv', split='test'), 
            test= dict(sample_rate = 0.1, json_filename = 'case_info_5souces.csv', split='test')
            )
            
model = dict(
        backbone = dict(verbose = False),
        neck = dict(upsample_cfg=dict(mode='trilinear', use_norm = True, 
                        kernel_size = (2,2,2), stride = (2,2,2)), 
                    verbose = False
                    ), 
        seg_head = dict(verbose = False), 
        mask_cfg = dict(mask_patch_size=32, mask_ratio=0.5), 
        verbose = False,

)

find_unused_parameters=True
load_from = 'work_dirs/simmim_convnext_s32c64_lung_192x192x160_100eps_interp/latest.pth'
resume_from = None  #'work_dirs/simmim_convnext_s32c64_lung_192x192x160_100eps_interp/latest.pth'

# optimizer
optimizer = dict(
                type='SGD', lr=0.01, momentum=0.9, weight_decay=1e-2, 
                # _delete_ = True, type='AdamW', lr=0.0002, weight_decay=0.0001
        ) 
optimizer_config = dict(_delete_ = True, grad_clip = dict(max_norm = 32, norm_type = 2)) # 31G
fp16 = dict(loss_scale = dict(init_scale=2**10, growth_factor=2.0, 
            backoff_factor=0.5, growth_interval=2000, enabled=True)) #30G
# learning policy
lr_config = dict(_delete_=True, 
                policy='poly', power=0.99, 
                #  policy='CosineAnnealing', min_lr=0.
                min_lr=1e-4, by_epoch=False,  warmup='linear', warmup_iters=1000, 
                 )

runner = dict(type='EpochBasedRunner', max_epochs=128)
checkpoint_config = dict(interval=2, max_keep_ckpts = 4)
# yapf:disable 
log_config = dict(interval=20, hooks=[
                dict(type='TextLoggerHook'), 
                # dict(type='TensorboardLoggerHook')
                ])


# CUDA_VISIBLE_DEVICES=0 python tools/train.py configs/selfup4med/simmim_convnext_s32c64_lung_192x192x160_100eps_interp.py --no-validate

# CUDA_VISIBLE_DEVICES=0,2,4 PORT=29024 bash ./tools/dist_train.sh configs/selfup4med/simmim_convnext_s32c64_lung_192x192x160_100eps_interp.py 3 --gpus 3 --no-validate
# CUDA_VISIBLE_DEVICES=0 python tools/test_med.py \
# configs/ribfrac/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn.py \
# work_dirs/retinanet3d_4l8c_vnet_3x_ribfrac_1cls_syncbn/latest.pth --eval recall  969798
