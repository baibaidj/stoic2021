_base_ = [
    '../datasets/stoic2021_lung_192x192x144.py',
]
# model settings
conv_cfg = dict(type = 'Conv3d')
norm_cfg = dict(type='IN3d', requires_grad=True) 
model = dict(
    type='ImageClassifierMed',
    backbone=dict(
        type='ConvNeXt3D',
        in_channels=1, 
        stem_cfg = dict(conv1kernel = 5, conv1stride = 2, conv1_chn_div = 1, 
                        ), 
        expand_ratio = 3, 
        dw_kernel_size = 5, 
        num_stages=4,
        depths=[0, 1, 3, 6, 9], 
        dims=[24, 24, 48, 96, 192],  # 2, 2, 4, 8, 16
        drop_path_rate=0.2, 
        layer_scale_init_value=0.1, 
        out_indices=(1, 2, 3, 4),
        conv_cfg=conv_cfg,
        norm_cfg=norm_cfg, 
        ), 
    head=dict(type='LinearClsHead',
            in_channels = 192,
            num_classes = 2,
            add_feat_dist = False, 
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