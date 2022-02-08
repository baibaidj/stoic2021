_base_ = [
    '../datasets/stoic2021_lung_192x192x160.py',
]
# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='IN3d', requires_grad=True) 
model = dict(
    type='ImageClassifierMed',
    backbone=dict(
        type='ConvNeXt3D',
        in_channels=1, 
        stem_cfg = dict(conv1stride = 4), 
        expand_ratio = 4, 
        dw_kernel_size = 7, 
        num_stages=5,
        depths=[0, 3, 3, 9, 3], 
        dims=[32, 32, 64, 128, 256],  # 2, 4, 8, 16, 32
        drop_path_rate=0.2, 
        layer_scale_init_value=1.0, 
        out_indices=(1, 2, 3, 4),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg, 
        ), 
    head=dict(type='LinearClsHead',
            in_channels = 256,
            num_classes = 1,
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
    target_class = 0, 
    test_cfg = None,
    train_cfg = None, 
)


    # detections_per_img = plan_arch.get("detections_per_img", 100)
    # score_thresh = plan_arch.get("score_thresh", 0)
    # topk_candidates = plan_arch.get("topk_candidates", 10000)
    # remove_small_boxes = plan_arch.get("remove_small_boxes", 0.01)
    # nms_thresh = plan_arch.get("nms_thresh", 0.6)