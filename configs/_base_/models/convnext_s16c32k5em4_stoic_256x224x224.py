_base_ = [
    '../datasets/stoic2021_lung_256x224x224.py',
]
# model settings
conv_cfg = dict(type = 'Conv3d')
#norm_cfg = dict(type='IN3d', requires_grad=True)  # IN3D
norm_cfg = dict(type='GN', num_groups=16, requires_grad=True) 
# dict(type='SyncBN', requires_grad=True) 
model = dict(
    type='ImageClassifierMed',
    backbone=dict(
        type='ConvNeXt3D',
        in_channels=1, 
        stem_cfg = dict(conv1stride = 4), 
        expand_ratio = 4, 
        dw_kernel_size = 7, 
        num_stages=4,
        depths=[0, 3, 3, 12], 
        dims=[32, 32, 64, 128],  # 4, 4, 8, 16
        drop_path_rate=0.2, 
        layer_scale_init_value=0.1, 
        out_indices=(1, 2, 3),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg, 
        ), 
    head=dict(type='LinearClsHead',
            in_channels = 128,
            num_classes = 2,
            add_feat_dist = False, 
            age_encoding= dict(),
            # is_multi_task = False, 
            loss=dict(type='CrossEntropyLoss', 
                        use_sigmoid = True, 
                        loss_weight=2.0, 
                        # class_weight = (0.8, 1.0)
                        ),
            in_index = -1, 
            dim = 3, dropout_ratio = 0.1), 

    gpu_aug_pipelines = {{ _base_.gpu_aug_pipelines }},
    train_cfg = dict(), 
    test_cfg = dict(),
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)